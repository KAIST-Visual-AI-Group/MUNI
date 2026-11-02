"""Import concrete source modules so their classes register once."""

from muni.data.sources import features as _features  # noqa: F401
from muni.data.sources import images as _images  # noqa: F401
from muni.data.sources import labels as _labels  # noqa: F401
from muni.data.sources import synthetic as _synthetic  # noqa: F401
