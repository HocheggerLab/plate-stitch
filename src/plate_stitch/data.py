"""Module for loading plate experiment data."""

import glob
import logging
import os
import re
import string
from functools import reduce


class PlateData:
    """Provides data loading of an Operetta plate experiment.

    The data is expected to be in the format of single plane tiff files
    named using the following convention:

        rNcNfNpN-chNskNfkNflN

    where: N is a number; r=row; c=column; f=field; p=plane (Z);
    ch=channel; sk=timepoint; fk=state; fl=Flim ID.
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
                "Some well positions have a different number of images: %s",
                wells,
            )

        self.well_positions: list[str] = [
            well_pos(r, c) for r, c in sorted(wells.keys())
        ]
        self.fields: list[int] = sorted(fields)
        self.planes: list[int] = sorted(planes)
        self.channels: list[int] = sorted(channels)
        self.times: list[int] = sorted(times)
        self.states: list[int] = sorted(states)
        self.flims: list[int] = sorted(flims)

    # TODO method to load an image as a numpy array TCZYX


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
