"""Module for exporting plate data."""

import os

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
    compression: str | None = None,
    overwrite: bool = False,
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
        overwrite: Overwrite existing masks.

    Raises:
        Exception if the plate has a z-stack; if the well position is invalid;
        or if the index selection is invalid.
    """
    # TODO: This does not check for a MaxIntensityProject (mip) for z-stacks.
    # For now just error.
    if len(plate.planes) > 1:
        raise Exception("Unsupported z-stack")

    # Control wells, timepoints, channels
    well_pos_list = plate.parseWells(wells)
    times_list = plate.parseTimes(times)
    channels_list = plate.parseChannels(channels)
    mask_channels_list = plate.parseMaskChannels(mask_channels)
    z = [1]  # Only support single plane stacks

    # Export each position as a time-series of TIFFs
    for well_pos in tqdm(well_pos_list, desc="Wells"):
        row, col = plate_pos(well_pos)
        basedir = os.path.join(outdir, str(well_pos))
        os.makedirs(basedir, exist_ok=True)

        for t in tqdm(times_list, desc=well_pos):
            # Images
            fn = os.path.join(basedir, f"i{t}.tif")
            if overwrite or not os.path.exists(fn):
                # Load N x TCZYX
                images = []
                for field in plate.fields:
                    images.append(
                        plate.get_image_data(
                            row, col, field, [t], channels_list, z
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
                _ = tifffile.imwrite(
                    fn,
                    stitched_images,
                    compression=compression,
                )

            if not len(mask_channels_list):
                continue

            # Labels
            fn = os.path.join(basedir, f"m{t}.tif")
            if overwrite or not os.path.exists(fn):
                # Load N x TCZYX
                labels = []
                for field in plate.fields:
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
                stitched_labels = stitch_labels(
                    np.array(labels).squeeze(axis=(1, 3)),
                    rotation=rotation,
                    overlap_x=overlap_x,
                    overlap_y=overlap_y,
                )
                _ = tifffile.imwrite(
                    fn,
                    stitched_labels,
                    compression=compression,
                )
