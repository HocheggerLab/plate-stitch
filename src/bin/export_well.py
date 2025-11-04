#!/usr/bin/env python3
"""Program to export a composed well image."""

import argparse
import logging
import os
import re

import numpy as np
import tifffile
from tqdm import tqdm

from plate_stitch.data import PlateData, plate_pos
from plate_stitch.stitching import stitch_images, stitch_labels
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

    group = parser.add_argument_group("Selection Options")
    group.add_argument(
        "--wells",
        default="All",
        help="Well positions (e.g. 'All; A1, A2') (default: %(default)s)",
    )
    group.add_argument(
        "--times",
        default="All",
        help="Time positions (e.g. All; 1,3; 2; 1,3) (default: %(default)s)",
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
        # TODO: This does not check for a MaxIntensityProject (mip) for z-stacks.
        # For now just error.
        if len(plate.planes) > 1:
            raise Exception("Unsupported z-stack")

        if args.wells == "All":
            well_pos_list = plate.well_positions
        else:
            well_pos_list = [item.strip() for item in args.wells.split(",")]
            for well_pos in well_pos_list:
                if well_pos not in plate.well_positions:
                    raise Exception(
                        f"Unknown well {well_pos} from {plate.well_positions}"
                    )

        # Control timepoints, channels
        time_points = _get_list(args.times, plate.times)
        channels = _get_list(args.channels, plate.channels)
        mask_channels = _get_list(args.mask_channels, plate.mask_channels)
        z = [1]  # Only support single plane stacks

        # Export each position as a series of Tiffs for a XYCZT hyperstack in ImageJ.
        logger.info("Exporting to %s", outdir)
        for well_pos in tqdm(well_pos_list, desc="Wells"):
            row, col = plate_pos(well_pos)
            basedir = os.path.join(outdir, str(well_pos))
            os.makedirs(basedir, exist_ok=True)

            for t in tqdm(time_points, desc=well_pos):
                images = []
                labels = []
                # Load N x TCZYX
                for field in plate.fields:
                    images.append(
                        plate.get_image_data(row, col, field, [t], channels, z)
                    )
                    if mask_channels:
                        labels.append(
                            plate.get_image_data(
                                row,
                                col,
                                field,
                                [t],
                                mask_channels,
                                z,
                                True,
                            )
                        )

                # Compose and save. Squeeze t and z axis from NTCZYX.
                stitched_images = stitch_images(
                    np.array(images).squeeze(axis=(1, 3)),
                    rotation=args.rotation,
                    overlap_x=args.ox,
                    overlap_y=args.oy,
                    edge=args.edge,
                    mode=args.mode,
                )
                tifffile.imwrite(
                    os.path.join(basedir, f"i{t}.tif"), stitched_images
                )
                if len(labels):
                    stitched_labels = stitch_labels(
                        np.array(labels).squeeze(axis=(1, 3)),
                        rotation=args.rotation,
                        overlap_x=args.ox,
                        overlap_y=args.oy,
                    )
                    tifffile.imwrite(
                        os.path.join(basedir, f"m{t}.tif"), stitched_labels
                    )

        logger.info("Export complete")


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
    if not (
        index.lower() == "all"
        or re.match(r"^(\d+(-\d+)?)(,\s*\d+(-\d+)?)*$", index)
    ):
        raise Exception(
            f"Selection input '{index}' doesn't match any of the expected patterns 'All, 1-3, 1'."
        )

    if index.lower() == "all":
        return x

    out = []
    for token in index.split(","):
        if "-" in index:
            # Handle range, e.g., '1-3'
            start, end = map(int, index.split("-"))
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


if __name__ == "__main__":
    main()
