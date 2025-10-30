#!/usr/bin/env python3
"""Program to list available plate data."""

import argparse
import glob
import os
import re

from plate_stitch.data import well_pos


def _dir_path(path: str) -> str:
    """Check if the path is a valid directory, or raise an error."""
    if os.path.isdir(path):
        return path
    else:
        raise FileNotFoundError(path)


# TODO: Convert this logic to a PlateDataset object. Constructor receives a directory.
# It provides attributes on the plate and the ability to load any row/col/field/t/c/z as numpy image TCZYX.


def main() -> None:
    """Program to list available plate data."""
    parser = argparse.ArgumentParser(
        description="""Program to list available plate data"""
    )
    parser.add_argument(
        "data", type=_dir_path, nargs="+", help="Plate data directory"
    )
    args = parser.parse_args()

    for dirname in args.data:
        print(dirname)
        tiff_files = list(glob.glob(os.path.join(dirname, "*.tiff")))
        pattern = re.compile(
            r"r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)sk(\d+)fk(\d+)fl(\d+)"
        )
        wells: dict[str, int] = {}
        fields = set()
        planes = set()
        channels = set()
        times = set()
        states = set()
        flims = set()
        for file in tiff_files:
            fn = os.path.basename(file)
            if m := pattern.match(fn):
                pos = well_pos(int(m.group(1)), int(m.group(2)))
                wells[pos] = wells.get(pos, 0) + 1
                fields.add(int(m.group(3)))
                planes.add(int(m.group(4)))
                channels.add(int(m.group(5)))
                times.add(int(m.group(6)))
                states.add(int(m.group(7)))
                flims.add(int(m.group(8)))

        # Simple check for complete data
        if len(set(wells.values())) != 1:
            print(
                "Some well positions have a different number of images:", wells
            )

        well_positions = list(wells)
        well_positions.sort()
        print("Well positions:", ", ".join(well_positions))
        for title, values in zip(
            ["Fields", "Planes", "Channels", "Times", "States", "FlimIDs"],
            [fields, planes, channels, times, states, flims],
            strict=False,
        ):
            v = list(values)
            v.sort()
            print(title + ":", ", ".join([str(x) for x in v]))


if __name__ == "__main__":
    main()
