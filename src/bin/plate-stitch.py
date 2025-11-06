#!/usr/bin/env python3
"""Program to segment and export a composed well image."""

import argparse

from plate_stitch.utils import dir_path


def main() -> None:
    """Program to segment and export a composed well image."""
    parser = argparse.ArgumentParser(
        description="""Program to segment and export a composed well image"""
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
    _ = group.add_argument(
        "--channels",
        default="All",
        help="Channels position (e.g. All; 1-3; 2; 1,3) (default: %(default)s)",
    )

    group = parser.add_argument_group("Correction Options")
    _ = group.add_argument(
        "--position-samples",
        type=int,
        default=100,
        help="Number of well positions to sample (default: %(default)s)",
    )
    _ = group.add_argument(
        "--time-samples",
        type=int,
        default=10,
        help="Number of time points to sample from each well position (default: %(default)s)",
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

    _ = group = parser.add_argument_group("Composition Options")
    _ = group.add_argument(
        "--rotation",
        default=0.15,
        help="Rotation angle in degrees counter clockwise (default: %(default)s)",
    )
    _ = group.add_argument(
        "--ox",
        default=7,
        help="Pixel overlap in x (default: %(default)s)",
    )
    _ = group.add_argument(
        "--oy",
        default=7,
        help="Pixel overlap in y (default: %(default)s)",
    )
    _ = group.add_argument(
        "--edge",
        default=7,
        help="Pixel edge for blending overlap (default: %(default)s)",
    )
    _ = group.add_argument(
        "--mode",
        default="reflect",
        help="Mode to fill points outside the image during rotation (default: %(default)s)",
    )

    group = parser.add_argument_group("Export Options")
    _ = group.add_argument(
        "--out",
        help="Output directory (defaults to the plate data directory)",
    )
    _ = group.add_argument(
        "--overwrite-export",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing export images (default: %(default)s)",
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
    from plate_stitch.export import export_wells
    from plate_stitch.flatfield import flatfield_correction
    from plate_stitch.segmentation import segment_nuclei

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    for dirname in args.data:
        logger.info(dirname)
        plate = PlateData(dirname)

        # Control wells, timepoints, channels
        wells = plate.parseWells(args.wells)
        times = plate.parseTimes(args.times)
        channels = plate.parseChannels(args.channels)
        # Currently only support nuclei segmentation (i.e. not cell segmentation)
        mask_channels = [args.nuclei_channel]

        # Note: Flat-field correction uses the entire plate (ignores wells; times; channels)
        logger.info("Creating flat-field correction")
        im = flatfield_correction(
            plate,
            positions=args.position_samples,
            time_points=args.time_samples,
        )
        logger.info("Correction image: %s %s", im.shape, im.dtype)

        logger.info("Segmenting nuclei channel: %d", args.nuclei_channel)
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
        logger.info("Segmentation complete")

        outdir = args.out if args.out else dirname
        logger.info("Exporting to %s", outdir)
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
            overwrite=args.overwrite_export,
        )
        logger.info("Export complete")


if __name__ == "__main__":
    main()
