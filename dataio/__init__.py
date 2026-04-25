# Dataset I/O — per-dataset loader subpackages.
#
# Renamed from ``datasets/`` to avoid shadowing the HuggingFace ``datasets``
# package on ``sys.path``. Each subpackage owns one dataset's on-disk format
# and exposes typed, dataclass-returning load helpers (e.g. ``dataio.sas``).
