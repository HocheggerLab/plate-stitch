"""Module for loading plate experiment data."""

import glob
import logging
import os
import re
import string
from functools import reduce
from typing import Any

import numpy as np
import numpy.typing as npt
from tifffile import imread


class PlateData:
    """Provides data loading of an Operetta plate experiment.

    The data is expected to be in the format of single plane tiff files
    named using the following convention:

        rNcNfNpN-chNskNfkNflN

    where: N is a number; r=row; c=column; f=field; p=plane (Z);
    ch=channel; sk=timepoint; fk=state; fl=Flim ID. Numbers prefixed
    by a single character are left padded with zeros to 2 digits wide,
    for example r01c01f01p01-ch1sk1fk1fl1.tiff.
    """

    def __init__(self, path: str) -> None:
        """Initialises the plate experiment.

        Args:
            path: Path to the plate images directory.
        """
        tiff_files = list(glob.glob(os.path.join(path, "*.tiff")))
        pattern = re.compile(
            r"r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)sk(\d+)fk(\d+)fl(\d+)"
        )
        wells: dict[tuple[int, int], int] = {}
        fields = set()
        planes = set()
        channels = set()
        times = set()
        states = set()
        flims = set()
        for file in tiff_files:
            fn = os.path.basename(file)
            if m := pattern.match(fn):
                pos = (int(m.group(1)), int(m.group(2)))
                wells[pos] = wells.get(pos, 0) + 1
                fields.add(int(m.group(3)))
                planes.add(int(m.group(4)))
                channels.add(int(m.group(5)))
                times.add(int(m.group(6)))
                states.add(int(m.group(7)))
                flims.add(int(m.group(8)))

        # Simple check for complete data
        if len(set(wells.values())) != 1:
            logging.getLogger(__name__).warning(
                "Some well positions directory '%s' have a different number of images: %s",
                path,
                wells,
            )

        self.path = path
        self.well_positions: list[str] = [
            well_pos(r, c) for r, c in sorted(wells.keys())
        ]
        self.fields: list[int] = sorted(fields)
        self.planes: list[int] = sorted(planes)
        self.channels: list[int] = sorted(channels)
        self.times: list[int] = sorted(times)
        self.states: list[int] = sorted(states)
        self.flims: list[int] = sorted(flims)

    def get_plane(
        self,
        row: int,
        col: int,
        field: int,
        t: int,
        c: int,
        z: int,
    ) -> npt.NDArray[Any]:
        """Load an image plane as a numpy array YX.

        Note: This method does not support the state or Flim ID identifiers
        in the image data filename. These are assumed to be 1.

        Args:
            row: Well row.
            col: Well column.
            field: Well field.
            t: Time.
            c: Channel.
            z: Z position.

        Returns:
            Image.
        """
        fn = os.path.join(
            self.path,
            f"r{row:02d}c{col:02d}f{field:02d}p{z:02d}-ch{c}sk{t}fk1fl1.tiff",
        )
        return imread(fn)

    def get_image(
        self,
        row: int,
        col: int,
        field: int,
        t: int,
        c: int,
        z: int,
        size_t: int = 1,
        size_c: int = 1,
        size_z: int = 1,
    ) -> npt.NDArray[Any]:
        """Load an image stack as a numpy array TCZYX.

        Note: This method does not support the state or Flim ID identifiers
        in the image data filename. These are assumed to be 1.

        Args:
            row: Well row.
            col: Well column.
            field: Well field.
            t: Start time.
            c: Start channel.
            z: Start Z position.
            size_t: Length of time points.
            size_c: Length of channels.
            size_z: Length of Z positions.

        Returns:
            Image.
        """
        data = []
        for tt in range(t, t + size_t):
            for cc in range(c, c + size_c):
                for zz in range(z, z + size_z):
                    data.append(self.get_plane(row, col, field, tt, cc, zz))
        return np.array(data).reshape((size_t, size_z, size_c) + data[0].shape)


# https://stackoverflow.com/a/48984697: convert-a-number-to-excel-s-base-26
def _from_excel(chars: str) -> int:
    return reduce(
        lambda r, x: r * 26 + x + 1,
        map(string.ascii_uppercase.index, chars),
        0,
    )


def _divmod_excel(n: int) -> tuple[int, int]:
    a, b = divmod(n, 26)
    if b == 0:
        return a - 1, b + 26
    return a, b


def _to_excel(num: int) -> str:
    chars = []
    while num > 0:
        num, d = _divmod_excel(num)
        chars.append(string.ascii_uppercase[d - 1])
    return "".join(reversed(chars))


def well_pos(row: int, col: int) -> str:
    """Convert plate row and column to plate well position.

    Uses a base 26 encoded row. This is coded to begin
    at A and ommits an explicit zero digit when incrementing
    the power of the base digit. Columns are labelled using
    the decimal representation. For example:
    1,1=A1; 2,1=B1; 2,2=B2; 26,1=Z1; 27,2=AA2.

    Args:
        row: Well row.
        col: Well column.

    Returns:
        Plate position.
    """
    # Delegate to a solution on StackOverflow
    return _to_excel(row) + str(col)


def plate_pos(well_pos: str) -> tuple[int, int]:
    """Convert plate well position to plate row and column.

    Uses a base 26 encoded row. This is coded to begin
    at A and ommits an explicit zero digit when incrementing
    the power of the base digit. Columns are labelled using
    the decimal representation. For example:
    1,1=A1; 2,1=B1; 2,2=B2; 26,1=Z1; 27,2=AA2.

    Args:
        well_pos: Well position.

    Returns:
        (row, column) position.

    Raises:
        Exception: if the format is not a recognised well position.
    """
    # Extract row and column
    if m := re.fullmatch(r"([A-Z]+)(\d+)", well_pos):
        # Delegate to a solution on StackOverflow
        return _from_excel(m.group(1)), int(m.group(2))
    raise Exception("Unknown plate well position: " + well_pos)
