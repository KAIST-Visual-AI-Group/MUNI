"""RGB and multi-view image data sources."""

import os

import torch
import torchvision.transforms.functional as TF

from muni.data.core import BaseSubdataset, register_subdataset
from muni.data.sources.image_cache import load_image_tensor_cache
from muni.util.crop import center_crop_arr
from muni.util.misc import print_on_main as print


_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _index_image_paths(root_dir: str, key_to_path: dict[str, str]) -> None:
    """Fill a recursive stem index, rejecting ambiguous keys in traversal order."""
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in _IMG_EXT:
                stem = os.path.splitext(fname)[0]
                path = os.path.join(dirpath, fname)
                previous = key_to_path.get(stem)
                if previous is not None:
                    raise ValueError(
                        f"Duplicate image stem {stem!r} under {root_dir}: "
                        f"{previous} and {path}. Dataset keys must be unique."
                    )
                key_to_path[stem] = path


@register_subdataset
class ImageSubdataset(BaseSubdataset):
    """Load RGB images from files or uint8 caches."""

    registry_name = "image"

    _CACHE: dict[
        tuple[object, ...], tuple[list[str], dict[str, int], torch.Tensor]
    ] = {}

    def __init__(
        self,
        root_dir: str | None = None,
        img_size: int = 256,
        cache_in_memory: bool = False,
        share_memory: bool = False,
        persistent_cache: str | None = None,
        cache_modality: str | None = None,
        mmap_cache: bool = False,
        labels_csv: str | None = None,
    ):
        self.img_size = int(img_size)
        self.cache_in_memory = bool(cache_in_memory) or persistent_cache is not None
        if self.cache_in_memory:
            self._load_uint8_cache(
                root_dir, share_memory, persistent_cache, cache_modality,
                mmap_cache, labels_csv,
            )
        else:
            self._key_to_path: dict[str, str] = {}
            _index_image_paths(root_dir, self._key_to_path)
            if not self._key_to_path:
                raise ValueError(f"No image files found under {root_dir}")
            self._keys = sorted(self._key_to_path)

    def _load_uint8_cache(
        self,
        root_dir: str | None,
        share_memory: bool,
        persistent_cache: str | None,
        cache_modality: str | None,
        mmap_cache: bool,
        labels_csv: str | None,
    ) -> None:
        self.root_dir = (
            os.path.abspath(os.path.expanduser(str(root_dir)))
            if root_dir is not None
            else None
        )
        self.persistent_cache = (
            os.path.abspath(os.path.expanduser(str(persistent_cache)))
            if persistent_cache is not None
            else None
        )
        modality = cache_modality or os.path.basename(os.path.normpath(self.root_dir))
        if self.persistent_cache is not None:
            payload = load_image_tensor_cache(
                self.persistent_cache, mmap=mmap_cache, labels_csv=labels_csv
            )
            if int(payload["img_size"]) != self.img_size:
                raise ValueError(
                    f"Persistent image cache img_size={payload['img_size']} does not match requested "
                    f"img_size={self.img_size}: {self.persistent_cache}"
                )
            modality_index = payload["modalities"].index(modality)
            self._keys = payload["keys"]
            self._key_to_index = {key: i for i, key in enumerate(self._keys)}
            self._cache_u8 = payload["images"][modality_index]
            return

        cache_key = ("decoded", self.root_dir, self.img_size)
        if cache_key not in self._CACHE:
            key_to_path: dict[str, str] = {}
            _index_image_paths(self.root_dir, key_to_path)
            keys = sorted(key_to_path)
            if not keys:
                raise ValueError(f"No image files found under {self.root_dir}")
            cache = self._build_uint8_cache(keys, key_to_path)
            if share_memory:
                cache.share_memory_()
            self._CACHE[cache_key] = (keys, {key: i for i, key in enumerate(keys)}, cache)
            cache_gib = cache.numel() * cache.element_size() / (1024**3)
            print(
                f"[CachedImageSubdataset] cached {len(keys)} images from "
                f"{self.root_dir} at {self.img_size}x{self.img_size} "
                f"({cache_gib:.2f} GiB uint8)"
            )
        self._keys, self._key_to_index, self._cache_u8 = self._CACHE[cache_key]

    def _build_uint8_cache(
        self,
        keys: list[str],
        key_to_path: dict[str, str],
    ) -> torch.Tensor:
        from PIL import Image

        cache = torch.empty(
            (len(keys), 3, self.img_size, self.img_size),
            dtype=torch.uint8,
        )
        for i, key in enumerate(keys):
            with Image.open(key_to_path[key]) as src:
                img = src.convert("RGB")
            img = center_crop_arr(img, self.img_size)
            cache[i].copy_(TF.pil_to_tensor(img))
        return cache

    def keys(self) -> list[str]:
        return self._keys

    def get(self, key: str, augment_state: dict | None = None) -> torch.Tensor:
        hflip = augment_state and augment_state.get("hflip")
        if self.cache_in_memory:
            tensor = self._cache_u8[self._key_to_index[key]]
            if hflip:
                tensor = tensor.flip(-1)
            tensor = tensor.to(torch.float32).div_(255.0).mul_(2.0).sub_(1.0)
        else:
            from PIL import Image

            with Image.open(self._key_to_path[key]) as source:
                img = source.convert("RGB")
            img = center_crop_arr(img, self.img_size)
            if hflip:
                img = TF.hflip(img)
            tensor = TF.pil_to_tensor(img).float().div_(255.0).mul_(2.0).sub_(1.0)
        return tensor


@register_subdataset
class MultiViewImageSubdataset(BaseSubdataset):
    """Load one fixed view per object from ``category/object/view.png`` trees."""

    registry_name = "multiview_image"

    def __init__(
        self, root_dir: str, view_index: int, img_size: int = 128, bg_color: float = 1.0
    ):
        self.img_size = img_size
        self.view_index = view_index
        self.bg_color = bg_color
        self._key_to_path: dict[str, str] = {}

        view_fname = f"{view_index:02d}.png"
        for cat_dir in sorted(os.listdir(root_dir)):
            cat_path = os.path.join(root_dir, cat_dir)
            if not os.path.isdir(cat_path):
                continue
            for obj_dir in sorted(os.listdir(cat_path)):
                view_file = os.path.join(cat_path, obj_dir, view_fname)
                if os.path.isfile(view_file):
                    self._key_to_path[f"{cat_dir}/{obj_dir}"] = view_file

        self._keys = sorted(self._key_to_path)

    def keys(self) -> list[str]:
        return self._keys

    def get(self, key: str, augment_state: dict | None = None) -> torch.Tensor:
        from PIL import Image

        with Image.open(self._key_to_path[key]) as source:
            if source.mode == "RGBA":
                bg_val = int(self.bg_color * 255)
                img = Image.new("RGB", source.size, (bg_val, bg_val, bg_val))
                img.paste(source, mask=source.split()[3])
            else:
                img = source.convert("RGB")
        img = center_crop_arr(img, self.img_size)
        return (TF.pil_to_tensor(img).float() / 255.0) * 2.0 - 1.0
