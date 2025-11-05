#!/usr/bin/env python3
"""Program to export a composed well image."""

import argparse
import logging

from plate_stitch.data import PlateData
from plate_stitch.export import export_wells
from plate_stitch.utils import dir_path


def main() -> None:
    """Program to export a composed well image."""
    parser = argparse.ArgumentParser(
        description="""Program to export a composed well image"""
    )

    parser.add_argument(
        "data", type=dir_path, nargs="+", help="Plate data directory"
    )
    parser.add_argument(
        "--out",
        help="Output directory (defaults to the plate data directory)",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing export images (default: %(default)s)",
    )

    group = parser.add_argument_group("Selection Options")
    group.add_argument(
        "--wells",
        default="All",
        help="Well positions (e.g. 'All; A1, A2') (default: %(default)s)",
    )
    group.add_argument(
        "--times",
        default="All",
        help="Time positions (e.g. All; 1-3; 2; 1,3) (default: %(default)s)",
    )
    group.add_argument(
        "--channels",
        default="All",
        help="Channels position (e.g. All; 1-3; 2; 1,3) (default: %(default)s)",
    )
    group.add_argument(
        "--mask-channels",
        default="All",
        help="Mask channels position (e.g. All; 1-3; 2; 1,3) (default: %(default)s)",
    )

    group = parser.add_argument_group("Composition Options")
    group.add_argument(
        "--rotation",
        default=0.15,
        help="Rotation angle in degrees counter clockwise (default: %(default)s)",
    )
    group.add_argument(
        "--ox",
        default=7,
        help="Pixel overlap in x (default: %(default)s)",
    )
    group.add_argument(
        "--oy",
        default=7,
        help="Pixel overlap in y (default: %(default)s)",
    )

    group = parser.add_argument_group("Image Options")
    group.add_argument(
        "--edge",
        default=7,
        help="Pixel edge for blending overlap (default: %(default)s)",
    )
    group.add_argument(
        "--mode",
        default="reflect",
        help="Mode to fill points outside the image during rotation (default: %(default)s)",
    )
    group.add_argument(
        "--compression",
        default="ZSTD",
        help="TIFF compression (e.g. None, LZW, ZSTD, ZLIB) (default: %(default)s)",
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
        outdir = args.out if args.out else dirname
        logger.info("Exporting to %s", outdir)

        # Control wells, timepoints, channels
        wells = plate.parseWells(args.wells)
        times = plate.parseTimes(args.times)
        channels = plate.parseChannels(args.channels)
        mask_channels = plate.parseMaskChannels(args.mask_channels)

        export_wells(
            plate,
            outdir,
            wells=wells,
            times=times,
            channels=channels,
            mask_channels=mask_channels,
            rotation=args.rotation,
            overlap_x=args.ox,
            overlap_y=args.oy,
            edge=args.edge,
            mode=args.mode,
            compression=args.compression,
            overwrite=args.overwrite,
        )

    logger.info("Export complete")


if __name__ == "__main__":
    main()
