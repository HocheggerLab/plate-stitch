"""Module for image segmentation functions."""

import logging
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from cellpose import models
from skimage import exposure
from skimage.segmentation import clear_border
from tifffile import imwrite
from tqdm import tqdm

from .data import PlateData, plate_pos
from .flatfield import load_flatfield_correction


def segment_nuclei(
    plate: PlateData,
    channel: int,
    wells: list[str] | None = None,
    times: list[int] | None = None,
    model_type: str = "Nuclei_Hoechst",
    diameter: float = 10,
    border: int = -1,
    overwrite: bool = False,
    device_name: str | None = None,
    compression: str | None = None,
) -> None:
    """Perform nuclei segmentation on the specified channel.

    Requires that the flatfield correction is computed.
    Objects close to the image border are optionally removed.

    Skips computation for existing segmentation images unless the overwrite flag
    is enabled.

    Args:
        plate: Plate data.
        channel: Nuclei channel.
        wells: Well positions.
        times: Time positions.
        model_type: Name of nuclei model.
        diameter: Expected nuclei diameter.
        border: Width of the border examined (negative to disable).
        overwrite: Overwrite existing masks.
        device_name: Torch device name (use None to auto-detect).
        compression: Compression for TIFF output files.

    Raises:
        Exception if the flat-field correction image is missing the nuclei channel.
    """
    ff_corr = load_flatfield_correction(plate)
    idx = plate.channels.index(channel)
    ff_corr = ff_corr[idx]

    # Cellpose 3
    device = _get_device(device_name)
    logging.info(
        "Segmenting nuclei using channel %d; device=%s", channel, device
    )
    model = models.CellposeModel(
        device=device,
        model_type=model_type,
    )
    n_channels = [[0, 0]]

    if wells is None:
        wells = plate.well_positions
    if times is None:
        times = plate.times

    total_ticks = len(plate.fields) * len(times) * len(plate.planes)
    logging.info(
        "Processing %d wells of %d images",
        len(wells),
        total_ticks,
    )
    for well_pos in tqdm(wells, desc="Wells"):
        with tqdm(total=total_ticks, desc=well_pos) as pbar:
            row, col = plate_pos(well_pos)
            for field in plate.fields:
                for t in times:
                    for z in plate.planes:
                        fn = plate.get_path(
                            row, col, field, t, channel, z, True
                        )
                        # Skip existing
                        if os.path.exists(fn) and not overwrite:
                            _ = pbar.update(1)
                            continue
                        # Apply flatfield correction and image scaling to the plane
                        im = plate.get_plane(row, col, field, t, channel, z)
                        im = im / ff_corr
                        im = _scale_img(im)
                        try:
                            mask, _, _ = model.eval(
                                im,
                                channels=n_channels,
                                diameter=diameter,
                                normalize=False,
                            )
                            mask = _filter_segmentation(mask, border=border)
                            mask = _compact_mask(mask)
                        except IndexError:
                            logging.warning(
                                "Failed create segmentation: %s", fn
                            )
                            # Create a dummy mask to allow identification of failures
                            mask = np.zeros((1, 1), dtype=np.uint8)
                        _ = imwrite(fn, mask, compression=compression)
                        _ = pbar.update(1)


def _get_device(name: str | None = None) -> torch.device:
    """Get a named torch device, or a default given the available backends."""
    if name is not None:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _scale_img(
    img: npt.NDArray[Any], percentile: tuple[float, float] = (1, 99)
) -> npt.NDArray[Any]:
    """Increase contrast by scaling image to exclude lowest and highest intensities.

    Args:
        img: Image
        percentile: Lower and upper range for intensities
    Returns:
        scaled image
    """
    percentiles = np.percentile(img, (percentile[0], percentile[1]))
    return exposure.rescale_intensity(img, in_range=tuple(percentiles))  # type: ignore


def _filter_segmentation(
    mask: npt.NDArray[Any], border: int = 5
) -> npt.NDArray[Any]:
    """Removes border objects and filters large and small objects from segmentation mask.

    The size of the border can be specified. Use a negative size to skip removal of
    border objects.

    Args:
        mask: unfiltered segmentation mask
        border: width of the border examined (negative to disable)

    Returns:
        filtered segmentation mask
    """
    cleared: npt.NDArray[Any] = (
        mask if border < 0 else clear_border(mask, buffer_size=border)  # type: ignore[no-untyped-call]
    )
    sizes = np.bincount(cleared.ravel())
    mask_sizes = sizes > 10
    mask_sizes[0] = 0
    cells_cleaned = mask_sizes[cleared]
    return cells_cleaned * mask  # type: ignore[no-any-return]


def _compact_mask(mask: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Compact the uint32 datatype to the smallest required to store all mask IDs.

    Args:
        mask (npt.NDArray[Any]): Segmentation mask.

    Returns:
        npt.NDArray[Any]: Compact segmentation mask.
    """
    m = mask.max()
    if m < 2**8:
        return mask.astype(np.uint8)
    if m < 2**16:
        return mask.astype(np.uint16)
    return mask
