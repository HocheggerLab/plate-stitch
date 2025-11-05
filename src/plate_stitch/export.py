"""Module for exporting plate data."""

import os
import re

import numpy as np
import tifffile
from tqdm import tqdm

from .data import PlateData, plate_pos
from .stitching import stitch_images, stitch_labels


def export_wells(
    plate: PlateData,
    outdir: str,
    wells: str = "All",
    times: str = "All",
    channels: str = "All",
    mask_channels: str = "All",
    rotation: float = 0,
    overlap_x: int = 0,
    overlap_y: int = 0,
    edge: int = 0,
    mode: str = "reflect",
    compression: tifffile.COMPRESSION | int | str | None = None,
) -> None:
    """Export the plate data as stitched images.

    Args:
        plate: Plate data.
        outdir: Output directory.
        wells: Well positions (e.g. 'All; A1, A2')
        times: Time positions (e.g. All; 1-3; 2; 1,3).
        channels: Channel positions (e.g. All; 1-3; 2; 1,3).
        mask_channels: Mask channel positions (e.g. All; 1-3; 2; 1,3).
        rotation: Rotation angle (degrees in counter-clockwise direction).
        overlap_x: Tile overlap in x.
        overlap_y: Tile overlap in y.
        edge: Edge size for blending overlaps.
        mode: Mode used to fill the rotated image outside the bounds
        (‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’).
        compression: Compression for TIFF output files.

    Raises:
        Exception if the plate has a z-stack; if the well position is invalid;
        or if the index selection is invalid.
    """
    # TODO: This does not check for a MaxIntensityProject (mip) for z-stacks.
    # For now just error.
    if len(plate.planes) > 1:
        raise Exception("Unsupported z-stack")

    if wells == "All":
        well_pos_list = plate.well_positions
    else:
        well_pos_list = [item.strip() for item in wells.split(",")]
        for well_pos in well_pos_list:
            if well_pos not in plate.well_positions:
                raise Exception(
                    f"Unknown well {well_pos} from {plate.well_positions}"
                )

    # Control timepoints, channels
    times_list = _get_list(times, plate.times)
    channels_list = _get_list(channels, plate.channels)
    mask_channels_list = _get_list(mask_channels, plate.mask_channels)
    z = [1]  # Only support single plane stacks

    # Export each position as a series of Tiffs for a XYCZT hyperstack in ImageJ.
    for well_pos in tqdm(well_pos_list, desc="Wells"):
        row, col = plate_pos(well_pos)
        basedir = os.path.join(outdir, str(well_pos))
        os.makedirs(basedir, exist_ok=True)

        for t in tqdm(times_list, desc=well_pos):
            images = []
            labels = []
            # Load N x TCZYX
            for field in plate.fields:
                images.append(
                    plate.get_image_data(
                        row, col, field, [t], channels_list, z
                    )
                )
                if mask_channels:
                    labels.append(
                        plate.get_image_data(
                            row,
                            col,
                            field,
                            [t],
                            mask_channels_list,
                            z,
                            True,
                        )
                    )

            # Compose and save. Squeeze t and z axis from NTCZYX.
            stitched_images = stitch_images(
                np.array(images).squeeze(axis=(1, 3)),
                rotation=rotation,
                overlap_x=overlap_x,
                overlap_y=overlap_y,
                edge=edge,
                mode=mode,
            )
            tifffile.imwrite(
                os.path.join(basedir, f"i{t}.tif"),
                stitched_images,
                compression=compression,
            )
            if len(labels):
                stitched_labels = stitch_labels(
                    np.array(labels).squeeze(axis=(1, 3)),
                    rotation=rotation,
                    overlap_x=overlap_x,
                    overlap_y=overlap_y,
                )
                tifffile.imwrite(
                    os.path.join(basedir, f"m{t}.tif"),
                    stitched_labels,
                    compression=compression,
                )


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
