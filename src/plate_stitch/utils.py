"""Module for utility functions."""

import os


def dir_path(path: str) -> str:
    """Check if the path is a valid directory, or raise an error.

    Args:
        path: Path.

    Returns:
        path

    Raises:
        FileNotFoundError if the path does not exist, or is not a directory.
    """
    if os.path.isdir(path):
        return path
    raise FileNotFoundError(path)
