"""Module for utility functions."""

import logging
import os
import random
from typing import Any

import numpy as np
import numpy.typing as npt
from tifffile import imread, imwrite

from .aggregator import ImageAggregator
from .data import PlateData, plate_pos


def flatfield_correction(
    plate: PlateData, positions: int = 100, time_points: int = 10
) -> npt.NDArray[Any]:
    """Load the flatfield correction for the given plate.

    If absent, the flatfield correction is computed using random sampling of
    image planes and saved to the plate directory.

    The returned image is CYX format. The channel index corresponds to the
    channel number at the corresponding index of the plate.channels data.

    Args:
        plate: Plate data.
        positions: Number of well positions to sample.
        time_points: Number of time points to sample from each well position.

    Returns:
        flatfield correction image (CYX).

    Raises:
        Exception if the plate images have a Z stack; or the existing flat-field correction
        image has the wrong number of channels.
    """
    if len(plate.planes) != 1:
        raise Exception("Z stacks are not supported")

    fn = os.path.join(plate.path, "flatfield.tiff")
    if os.path.exists(fn):
        im = imread(fn)
        if im.shape[0] != len(plate.channels):
            raise Exception(
                f"Flat-field correction image '{fn}' has incorrect number of channels"
            )
        return im

    logger = logging.getLogger(__name__)

    agg = [ImageAggregator() for _ in plate.channels]

    # Sample from the well fields
    nwells = len(plate.well_positions)
    nfields = len(plate.fields)
    ntimes = len(plate.times)
    total = nwells * nfields
    samples = (
        list(range(total))
        if total < positions
        else random.sample(range(total), positions)
    )
    for s in samples:
        well_pos, field = divmod(s, nfields)
        row, col = plate_pos(plate.well_positions[well_pos])
        field = plate.fields[field]
        # Sample timepoints
        time_samples = (
            list(range(ntimes))
            if ntimes < time_points
            else random.sample(range(ntimes), time_points)
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Sampling %s field=%d t=%s",
                plate.well_positions[well_pos],
                field,
                [plate.times[x] for x in time_samples],
            )
        for time in time_samples:
            t = plate.times[time]
            for i, c in enumerate(plate.channels):
                # Do not support z stacks. This could be done using a maximum intensity
                # projection to sample intensity evenly across the image field.
                agg[i].add_image(plate.get_plane(row, col, field, t, c, 1))

    imgs = []
    for i, a in enumerate(agg):
        blurred_agg_img = a.get_gaussian_image(30)
        assert blurred_agg_img is not None, (
            "Failed to aggregate channel: " + str(plate.channels[i])
        )
        norm_img: npt.NDArray[Any] = blurred_agg_img / blurred_agg_img.mean()
        imgs.append(norm_img)

    im = np.array(imgs)
    imwrite(fn, im)
    return im
