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
            r"r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)sk(\d+)fk(\d+)fl(\d+)(.tiff|-mask.tiff)"
        )
        wells: dict[tuple[int, int], int] = {}
        fields = set()
        planes = set()
        channels = set()
        mask_channels = set()
        times = set()
        states = set()
        flims = set()
        for file in tiff_files:
            fn = os.path.basename(file)
            if m := pattern.fullmatch(fn):
                pos = (int(m.group(1)), int(m.group(2)))
                wells[pos] = wells.get(pos, 0) + 1
                fields.add(int(m.group(3)))
                planes.add(int(m.group(4)))
                if m.group(9) == ".tiff":
                    channels.add(int(m.group(5)))
                else:
                    mask_channels.add(int(m.group(5)))
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
        self.mask_channels: list[int] = sorted(mask_channels)
        self.times: list[int] = sorted(times)
        self.states: list[int] = sorted(states)
        self.flims: list[int] = sorted(flims)

    def get_path(
        self,
        row: int,
        col: int,
        field: int,
        t: int,
        c: int,
        z: int,
        mask: bool = False,
    ) -> str:
        """Get the full path of an image plane.

        Note: This method does not support the state or Flim ID identifiers
        in the image data filename. These are assumed to be 1.

        Args:
            row: Well row.
            col: Well column.
            field: Well field.
            t: Time.
            c: Channel.
            z: Z position.
            mask: True to return the path for the image mask.

        Returns:
            Path.
        """
        suffix = "-mask.tiff" if mask else ".tiff"
        return os.path.join(
            self.path,
            f"r{row:02d}c{col:02d}f{field:02d}p{z:02d}-ch{c}sk{t}fk1fl1{suffix}",
        )

    def get_plane(
        self,
        row: int,
        col: int,
        field: int,
        t: int,
        c: int,
        z: int,
        mask: bool = False,
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
            mask: True to return the image mask.

        Returns:
            Image.
        """
        return imread(self.get_path(row, col, field, t, c, z, mask))

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
        mask: bool = False,
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
            mask: True to return the image mask.

        Returns:
            Image.
        """
        data = []
        for tt in range(t, t + size_t):
            for cc in range(c, c + size_c):
                for zz in range(z, z + size_z):
                    data.append(
                        self.get_plane(row, col, field, tt, cc, zz, mask)
                    )
        return np.array(data).reshape((size_t, size_c, size_z) + data[0].shape)

    def get_image_data(
        self,
        row: int,
        col: int,
        field: int,
        t: list[int],
        c: list[int],
        z: list[int],
        mask: bool = False,
    ) -> npt.NDArray[Any]:
        """Load image data as a numpy array TCZYX.

        This method support discontinuous data ranges.

        Note: This method does not support the state or Flim ID identifiers
        in the image data filename. These are assumed to be 1.

        Args:
            row: Well row.
            col: Well column.
            field: Well field.
            t: Time points.
            c: Channels.
            z: Z positions.
            mask: True to return the image mask.

        Returns:
            Image.
        """
        data = []
        for tt in t:
            for cc in c:
                for zz in z:
                    data.append(
                        self.get_plane(row, col, field, tt, cc, zz, mask)
                    )
        return np.array(data).reshape((len(t), len(c), len(z)) + data[0].shape)

    def parseWells(self, value: str) -> list[str]:
        """Get a list of wells from the plate.

        Args:
            value: Selection value (e.g. All; A1, A2).

        Returns:
            list

        Raises:
            Exception if the well is not in the plate.
        """
        if value == "All":
            return self.well_positions.copy()
        well_pos_list = [item.strip() for item in value.split(",")]
        for well_pos in well_pos_list:
            if well_pos not in self.well_positions:
                raise Exception(
                    f"Unknown well {well_pos} from {self.well_positions}"
                )
        return sorted(well_pos_list)

    def parseTimes(self, value: str) -> list[int]:
        """Get a list of times from the plate.

        Args:
            value: Selection value (e.g. All; 1-3; 2; 1,3).

        Returns:
            list

        Raises:
            Exception if the selection is unrecognised or is not in the plate.
        """
        return _get_list(value, self.times)

    def parseChannels(self, value: str) -> list[int]:
        """Get a list of channels from the plate.

        Args:
            value: Selection value (e.g. All; 1-3; 2; 1,3).

        Returns:
            list

        Raises:
            Exception if the selection is unrecognised or is not in the plate.
        """
        return _get_list(value, self.channels)

    def parseMaskChannels(self, value: str) -> list[int]:
        """Get a list of mask channels from the plate.

        Args:
            value: Selection value (e.g. All; 1-3; 2; 1,3).

        Returns:
            list

        Raises:
            Exception if the selection is unrecognised or is not in the plate.
        """
        return _get_list(value, self.mask_channels)


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


def _get_list(index: str, x: list[int]) -> list[int]:
    """Get a selection of values from a list.

    Supports returning the entire list using 'All'; known values
    from the list as comma delimited values; or a range '1-3'.

    Args:
        index: Selection value.
        x: List

    Returns:
        Selecion of values.
    """
    is_all = index.lower() == "all"
    if not (is_all or re.match(r"^(\d+(-\d+)?)(,\s*\d+(-\d+)?)*$", index)):
        raise Exception(
            f"Selection input '{index}' doesn't match any of the expected patterns 'All, 1-3, 1'."
        )

    if is_all:
        return x

    out = []
    for token in index.split(","):
        token = token.rstrip()
        if "-" in token:
            # Handle range, e.g., '1-3'
            start, end = map(int, token.split("-"))
            out.extend(list(range(start, end + 1)))
        else:
            # Handle single number, e.g., '1'
            out.append(int(token))

    out = sorted(set(out))
    for v in out:
        if v not in x:
            raise Exception(
                f"Unknown index [{v}] from selection {index} using list: {x}"
            )
    return out
