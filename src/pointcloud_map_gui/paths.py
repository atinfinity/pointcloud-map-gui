"""Where the repository's data lives.

The sample clouds are 19 MB and belong to the repository, not to the
installed package, so they are found by walking up from this file. That holds
for the editable install `uv sync` produces, which is how this project is
meant to be run; a wheel copied somewhere else has no repository above it, and
the callers raise rather than quietly reading from the wrong place.
"""
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent


def repository_root():
    """The checkout this package was installed from, or None."""
    root = _PACKAGE_ROOT.parents[1]  # src/pointcloud_map_gui -> src -> repo
    return root if (root / "pyproject.toml").is_file() else None


def sample_data_dir():
    """The bundled sample clouds, or None when running outside a checkout."""
    root = repository_root()
    if root is None:
        return None
    directory = root / "sample_data"
    return directory if directory.is_dir() else None


def require_sample_data_dir():
    directory = sample_data_dir()
    if directory is None:
        raise FileNotFoundError(
            "sample_data/ is only available when running from a checkout of the "
            "project; pass an explicit path instead"
        )
    return directory
