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
    wells: list[str] | None = None,
    times: list[int] | None = None,
    channels: list[int] | None = None,
    mask_channels: list[int] | None = None,
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
        wells: Well positions.
        times: Time positions.
        channels: Channel positions.
        mask_channels: Mask channel positions.
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

    z = [1]  # Only support single plane stacks

    if wells is None:
        wells = plate.well_positions
    if times is None:
        times = plate.times
    if channels is None:
        channels = plate.channels
    if mask_channels is None:
        mask_channels = plate.mask_channels

    # Export each position as a time-series of TIFFs
    for well_pos in tqdm(wells, desc="Wells"):
        row, col = plate_pos(well_pos)
        basedir = os.path.join(outdir, str(well_pos))
        os.makedirs(basedir, exist_ok=True)

        for t in tqdm(times, desc=well_pos):
            # Images
            fn = os.path.join(basedir, f"i{t}.tif")
            if overwrite or not os.path.exists(fn):
                # Load N x TCZYX
                images = []
                for field in plate.fields:
                    images.append(
                        plate.get_image_data(row, col, field, [t], channels, z)
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

            if not len(mask_channels):
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
                            mask_channels,
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
