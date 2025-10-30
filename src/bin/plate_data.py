#!/usr/bin/env python3
"""Program to list available plate data."""

import argparse
import logging
import os

from plate_stitch.data import PlateData


def _dir_path(path: str) -> str:
    """Check if the path is a valid directory, or raise an error."""
    if os.path.isdir(path):
        return path
    else:
        raise FileNotFoundError(path)


def main() -> None:
    """Program to list available plate data."""
    parser = argparse.ArgumentParser(
        description="""Program to list available plate data"""
    )
    parser.add_argument(
        "data", type=_dir_path, nargs="+", help="Plate data directory"
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s", level=logging.INFO
    )

    for dirname in args.data:
        logger.info(dirname)
        plate = PlateData(dirname)
        out = []
        for title, values in zip(
            [
                "Wells",
                "Fields",
                "Planes",
                "Channels",
                "Times",
                "States",
                "FlimIDs",
            ],
            [
                plate.well_positions,
                plate.fields,
                plate.planes,
                plate.channels,
                plate.times,
                plate.states,
                plate.flims,
            ],
            strict=True,
        ):
            v: list[str | int] = values  # type: ignore[assignment]
            txt = ", ".join([str(x) for x in v])
            out.append(f"{title:9s}: {txt}")
        print("\n".join(out))


if __name__ == "__main__":
    main()
