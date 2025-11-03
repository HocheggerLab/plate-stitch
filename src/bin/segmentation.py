#!/usr/bin/env python3
"""Program to generate segmentation images for plate data."""

import argparse
import logging

from plate_stitch.data import PlateData
from plate_stitch.segmentation import segment_nuclei
from plate_stitch.utils import dir_path


def main() -> None:
    """Program to generate segmentation images for plate data."""
    parser = argparse.ArgumentParser(
        description="""Program to generate segmentation images for plate data"""
    )
    parser.add_argument(
        "data", type=dir_path, nargs="+", help="Plate data directory"
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=1,
        help="Nuclei channel (default: %(default)s)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="Nuclei_Hoechst",
        help="Name of nuclei model (default: %(default)s)",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=10,
        help="Expected nuclei diameter (pixels) (default: %(default)s)",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=-1,
        help="Width of the border to exclude border objects (negative to disable; default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing masks (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Torch device name (default: auto-detect)",
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    for dirname in args.data:
        logger.info(dirname)
        plate = PlateData(dirname)
        segment_nuclei(
            plate,
            args.channel,
            model_type=args.model_type,
            diameter=args.diameter,
            border=args.border,
            overwrite=args.overwrite,
            device_name=args.device,
        )


if __name__ == "__main__":
    main()
