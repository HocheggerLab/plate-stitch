#!/usr/bin/env python3
"""Program to generate flat-field correction masks for plate data."""

import argparse
import logging

from plate_stitch.data import PlateData
from plate_stitch.flatfield import flatfield_correction
from plate_stitch.utils import dir_path


def main() -> None:
    """Program to generate flat-field correction masks for plate data."""
    parser = argparse.ArgumentParser(
        description="""Program to generate flat-field correction masks for plate data"""
    )
    parser.add_argument(
        "data", type=dir_path, nargs="+", help="Plate data directory"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use debug logging (default: %(default)s)",
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=100,
        help="Number of well positions to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--time_points",
        type=int,
        default=10,
        help="Number of time points to sample from each well position (default: %(default)s)",
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    for dirname in args.data:
        logger.info(dirname)
        plate = PlateData(dirname)
        im = flatfield_correction(
            plate, positions=args.positions, time_points=args.time_points
        )
        logger.info("Correction image: %s %s", im.shape, im.dtype)


if __name__ == "__main__":
    main()
