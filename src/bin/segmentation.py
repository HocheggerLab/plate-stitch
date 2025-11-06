#!/usr/bin/env python3
"""Program to generate segmentation images for plate data."""

import argparse

from plate_stitch.utils import dir_path


def main() -> None:
    """Program to generate segmentation images for plate data."""
    parser = argparse.ArgumentParser(
        description="""Program to generate segmentation images for plate data"""
    )
    _ = parser.add_argument(
        "data", type=dir_path, nargs="+", help="Plate data directory"
    )

    group = parser.add_argument_group("Selection Options")
    _ = group.add_argument(
        "--wells",
        default="All",
        help="Well positions (e.g. 'All; A1, A2') (default: %(default)s)",
    )
    _ = group.add_argument(
        "--times",
        default="All",
        help="Time positions (e.g. All; 1-3; 2; 1,3) (default: %(default)s)",
    )

    group = parser.add_argument_group("Segmentation Options")
    _ = group.add_argument(
        "--nuclei-channel",
        type=int,
        default=1,
        help="Nuclei channel (default: %(default)s)",
    )
    _ = group.add_argument(
        "--model-type",
        type=str,
        default="Nuclei_Hoechst",
        help="Name of nuclei model (default: %(default)s)",
    )
    _ = group.add_argument(
        "--diameter",
        type=float,
        default=10,
        help="Expected nuclei diameter (pixels) (default: %(default)s)",
    )
    _ = group.add_argument(
        "--border",
        type=int,
        default=-1,
        help="Width of the border to exclude border objects (negative to disable; default: %(default)s)",
    )
    _ = group.add_argument(
        "--overwrite-masks",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing masks (default: %(default)s)",
    )
    _ = group.add_argument(
        "--device",
        type=str,
        help="Torch device name (default: auto-detect)",
    )

    group = parser.add_argument_group("Image Options")
    _ = group.add_argument(
        "--compression",
        default="ZSTD",
        help="TIFF compression (e.g. None, LZW, ZSTD, ZLIB) (default: %(default)s)",
    )

    args = parser.parse_args()

    # Delay imports until argument parsing succeeds
    import logging

    from plate_stitch.data import PlateData
    from plate_stitch.segmentation import segment_nuclei

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    for dirname in args.data:
        logger.info(dirname)
        plate = PlateData(dirname)

        # Control wells, timepoints
        wells = plate.parseWells(args.wells)
        times = plate.parseTimes(args.times)

        segment_nuclei(
            plate,
            args.nuclei_channel,
            wells=wells,
            times=times,
            model_type=args.model_type,
            diameter=args.diameter,
            border=args.border,
            overwrite=args.overwrite_masks,
            device_name=args.device,
            compression=args.compression,
        )


if __name__ == "__main__":
    main()
