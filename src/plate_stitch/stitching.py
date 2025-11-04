"""Module for stitching well samples into a single image."""

import logging
import math
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.ndimage
import skimage.transform as transform
from skimage.util import map_array

logger = logging.getLogger(__name__)


def stitch_images(
    images: npt.NDArray[Any],
    rotation: float = 0.0,
    overlap_x: int = 0,
    overlap_y: int = 0,
    edge: int = 0,
    mode: str = "reflect",
) -> npt.NDArray[Any]:
    """Stitch the images in the array.

    Uses the specified pattern when a full well is imaged at 10x on an Operetta microscope.

    Supports:
    5x5 grid with corners excluded which creates 21 images; 2x2 grid of 4 images;
    3x3 grid of 9 images; 1x2 grid of 2 images.

    Args:
        images: Input image N-images of (CYX).
        rotation: Rotation angle (degrees in counter-clockwise direction).
        overlap_x: Tile overlap in x.
        overlap_y: Tile overlap in y.
        edge: Edge size for blending overlaps.
        mode: Mode used to fill the rotated image outside the bounds
        (‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’).

    Returns:
        Stitched image
    """
    logger.debug("Stitching images %s", images.shape)
    # NCYX
    size = len(images.shape)
    assert size == 4, "The input array should be N-images of CYX"
    n = images.shape[0]
    indices_pattern = _get_stitch_pattern(n)

    # YX order
    tiles: dict[int, dict[int, npt.NDArray[Any]]] = {}
    for y, row in enumerate(indices_pattern):
        for x, idx in enumerate(row):
            if idx != -1:
                d = tiles.get(x)
                if not d:
                    tiles[x] = d = {}
                d[y] = images[idx]

    stitched_image = _compose_tiles(
        tiles,
        rotation=rotation,
        ox=-overlap_x,
        oy=-overlap_y,
        edge=edge,
        mode=mode,
    )

    logger.debug("Stitched image shape: %s", stitched_image.shape)
    return stitched_image


def _get_stitch_pattern(n: int) -> list[list[int]]:
    """Gets the Operetta microscope stitch pattern for the given number of tiles.

    Args:
        n: Number of tiles

    Returns:
        list of lists of indices: rows of y; each row contains columns of x. The index
        at the YX position is the index into the n tiles for that (X,Y) location.
    """
    if n == 21:
        indices_pattern = [
            [-1, 1, 2, 3, -1],  # Preserved -1 for empty
            [8, 7, 6, 5, 4],  # Adjusted for zero-based indexing
            [9, 10, 0, 11, 12],  # The first image is now 0 (zero-based)
            [17, 16, 15, 14, 13],  # Adjusted for zero-based indexing
            [-1, 18, 19, 20, -1],  # Preserved -1 for empty
        ]
    elif n == 4:
        indices_pattern = [
            [1, 2],
            [3, 0],
        ]
    elif n == 2:
        indices_pattern = [
            [0, 1],
        ]
    elif n == 9:
        indices_pattern = [
            [1, 2, 3],
            [6, 5, 4],
            [7, 8, 0],
        ]
    else:
        raise ValueError(f"Unsupported number of image tiles: {n}")
    return indices_pattern


def _compose_tiles(
    tiles: dict[int, dict[int, npt.NDArray[Any]]],
    rotation: float = 0,
    ox: int = 0,
    oy: int = 0,
    edge: int = 0,
    mode: str = "reflect",
) -> npt.NDArray[Any]:
    """Compose tiles into a single image.

    It is assumed all tiles are the same shape: CYX.

    Args:
        tiles: Dictionary of dictionaries of tiles, keyed by [x][y].
        rotation: Rotation angle (degrees in counter-clockwise direction).
        ox: Tile offset in x (use negative for overlap).
        oy: Tile offset in y (use negative for overlap).
        edge: Edge size for blending overlaps.
        mode: Mode used to fill the rotated image outside the bounds
        (‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’).

    Returns:
        composed (np.array): The composed image (CYX).
    """
    # Compute tile grid dimensions
    maxx = np.max(list(tiles.keys()))
    maxy = 0
    for d in tiles.values():
        maxy = np.max(list(d.keys()), initial=maxy)

    # Create rotated mask using the first image to set the dimensions.
    # The mask uses nearest-neighbour interpolation to effectivly mark the pixels
    # of interest.
    y = next(iter(tiles[maxx]))
    im = tiles[maxx][y]
    original_shape = im.shape
    rotated_mask = transform.rotate(
        np.ones(original_shape[-2:], dtype=int),
        rotation,
        resize=True,
        preserve_range=True,
        order=0,
    )
    new_shape = rotated_mask.shape

    # preserve image type
    dtype = im.dtype

    # Create weights for blending overlap.
    if edge:
        # Distance transform does not use out-of-bounds as background.
        # So pad with 1 pixel and crop.
        edt = scipy.ndimage.distance_transform_edt(np.pad(rotated_mask, 1))
        edt = edt[1:-1, 1:-1]
        edt = np.clip(edt, a_min=0, a_max=edge)
        rotated_mask = edt / edge

    # Create output.
    # Note that arrays are CYX format.
    channels = original_shape[0]
    out = np.zeros(
        (
            channels,
            (maxy + 1) * new_shape[0] + maxy * oy,
            (maxx + 1) * new_shape[1] + maxx * ox,
        )
    )
    total = np.zeros(out.shape[-2:])

    # Rotate each image and insert
    for x, d in tiles.items():
        for y, im in d.items():
            imgs = []
            for c in range(channels):
                # Multiply the rotation by the mask (which optionally weights pixels).
                # The rotation uses bilinear interpolation with edge-pixel extension to generate
                # reasonable intensity edge pixels. The mode can be varied.
                imgs.append(
                    rotated_mask
                    * transform.rotate(
                        im[c, ...],
                        rotation,
                        resize=True,
                        preserve_range=True,
                        order=1,
                        mode=mode,
                    )
                )
            # Original shape sets the translation
            xp = x * (original_shape[2] + ox)
            yp = y * (original_shape[1] + oy)
            # New shape defines the range of the rotation image.
            # Note that arrays are YX format.
            out[:, yp : yp + new_shape[0], xp : xp + new_shape[1]] += np.stack(
                imgs
            )
            total[yp : yp + new_shape[0], xp : xp + new_shape[1]] += (
                rotated_mask
            )

    indices = total != 0
    for c in range(channels):
        out[c, ...] = np.divide(
            out[c, ...], total, where=indices, out=np.zeros(total.shape)
        )
    return _as_dtype(dtype, out)


def _as_dtype(
    dtype: np.dtype[Any], array: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """Return the result array as the given type.

    The array is clipped in-place to the min/max range for the type and a new array returned.

    Args:
        dtype: Desired type.
        array: Array.

    Returns:
        array converted to the type
    """
    # Retain type
    if issubclass(dtype.type, np.integer):
        info = np.iinfo(dtype)
        return np.clip(
            array, a_min=info.min, a_max=info.max, out=array
        ).astype(dtype)
    if issubclass(dtype.type, np.floating):
        info = np.finfo(dtype)
        return np.clip(
            array, a_min=info.min, a_max=info.max, out=array
        ).astype(dtype)
    # Possibly not a correct dtype?
    return array


def stitch_labels(
    labels: npt.NDArray[Any],
    rotation: float = 0.0,
    overlap_x: int = 0,
    overlap_y: int = 0,
) -> npt.NDArray[Any]:
    """Stitch the labels in the array.

    Uses the specified pattern when a full well is imaged at 10x on an Operetta microscope.

    Supports:
    5x5 grid with corners excluded which creates 21 images; 2x2 grid of 4 images;
    3x3 grid of 9 images; 1x2 grid of 2 images.

    Note: labels will be renumberd to unique objects.

    Args:
        labels: Input label image N-images of (CYX).
        rotation: Rotation angle (degrees in counter-clockwise direction).
        overlap_x: Tile overlap in x.
        overlap_y: Tile overlap in y.
        edge: Edge size for blending overlaps.
        mode: Mode used to fill the rotated image outside the bounds
        (‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’).

    Returns:
        Stitched labels
    """
    logger.debug("Stitching labels %s", labels.shape)
    # NCYX
    size = len(labels.shape)
    assert size == 4, "The input array should be N-images of CYX"
    n = labels.shape[0]
    indices_pattern = _get_stitch_pattern(n)

    tiles: dict[int, dict[int, npt.NDArray[Any]]] = {}
    for y, row in enumerate(indices_pattern):
        for x, idx in enumerate(row):
            if idx != -1:
                d = tiles.get(x)
                if not d:
                    tiles[x] = d = {}
                d[y] = labels[idx]

    stitched_image = _compose_labels(
        tiles, rotation=rotation, ox=-overlap_x, oy=-overlap_y
    )

    logger.debug("Stitched labels shape: %s", stitched_image.shape)
    return stitched_image


def _compose_labels(
    tiles: dict[int, dict[int, npt.NDArray[Any]]],
    rotation: float = 0,
    ox: int = 0,
    oy: int = 0,
) -> npt.NDArray[Any]:
    """Compose labels tiles into a single image.

    It is assumed all tiles are the same shape: YXC.
    The unique ID of labels will be remapped. Overlapping labels on adjacent tiles are mapped
    to the same ID.

    Args:
        tiles (dict): Dictionary of dictionaries of tiles, keyed by [x][y].
        rotation (float): Rotation angle (degrees in counter-clockwise direction).
        ox (int): Tile offset in x (use negative for overlap).
        oy (int): Tile offset in y (use negative for overlap).

    Returns:
        The composed labels (CYX).
    """
    # Compute tile grid dimensions
    maxx = np.max(list(tiles.keys()))
    maxy = 0
    for d in tiles.values():
        maxy = np.max(list(d.keys()), initial=maxy)

    # Rotate the first image to set the dimensions.
    y = next(iter(tiles[maxx]))
    original_shape = tiles[maxx][y].shape
    new_shape = transform.rotate(
        np.ones(original_shape[-2:], dtype=int),
        rotation,
        resize=True,
        preserve_range=True,
        order=0,
    ).shape

    # Create output.
    # Note that arrays are CYX format.
    channels = original_shape[0]
    out = [
        np.zeros(
            (
                (maxy + 1) * new_shape[0] + maxy * oy,
                (maxx + 1) * new_shape[1] + maxx * ox,
            ),
            dtype=tiles[maxx][y].dtype,
        )
        for i in range(channels)
    ]

    border = 0
    if ox < 0:
        border = -ox
    if oy < 0:
        border = max(border, -oy)

    # Rotate each image and insert
    for x, d in tiles.items():
        for y, im in d.items():
            # Original shape sets the translation
            xp = x * (original_shape[2] + ox)
            yp = y * (original_shape[1] + oy)
            for c in range(channels):
                # The rotation uses nearest neighbour interpolation to maintain IDs.
                rotated = transform.rotate(
                    im[c, ...],
                    rotation,
                    resize=True,
                    preserve_range=True,
                    order=0,
                )
                out[c] = merge_labels(
                    out[c], rotated, xp=xp, yp=yp, border=border
                )

    return np.stack(out)


def merge_labels(
    im1: npt.NDArray[Any],
    im2: npt.NDArray[Any],
    xp: int = 0,
    yp: int = 0,
    border: int = 0,
) -> npt.NDArray[Any]:
    """Merges the labels in image 2 into image 1.

    Image 2 may be smaller than image 1.
    Scans pixels in the border against the current labels. Any overlapping labels
    in the new image adopt the ID of the overlapping label.

    Args:
        im1: Current labels.
        im2: New labels.
        xp: Offset in x.
        yp: Offset in y.
        border: Border width.

    Returns:
        updated (np.array): The updated labels.
    """
    s = im2.shape
    # Avoid overlap analysis when no border or all-zero current image
    if not (border and im1.any()):
        return _merge_nonoverlapping_labels(im1, im2, xp=xp, yp=yp)

    # Extract current sub-image
    im1a = im1[yp : yp + s[0], xp : xp + s[1]]
    # Overlap mask
    overlap = (im1a != 0) & (im2 != 0)
    if not overlap.any():
        return _merge_nonoverlapping_labels(im1, im2, xp=xp, yp=yp)

    # Count size of label overlaps in border
    m1 = im1a.reshape(-1) * overlap.reshape(-1)
    m2 = im2.reshape(-1) * overlap.reshape(-1)
    h1o = np.bincount(m1)
    h2o = np.bincount(m2)
    # Require a new -> old ID overlap histogram.
    # Assume new IDs are sequential from 1.
    # Remap old IDs that are in the overlap from 1 to save memory.
    id_map = np.zeros(len(h1o), dtype=np.uint16)
    reverse_map = np.zeros(len(h1o), dtype=np.uint16)
    new_id = 0
    for i, c in enumerate(h1o):
        if c:
            id_map[i] = new_id
            reverse_map[new_id] = i
            new_id += 1
    # TODO: This can possibly be sped up using bincount
    h = np.zeros((np.nonzero(h2o)[0][-1] + 1, new_id), dtype=np.uint16)
    for a, b in zip(m2, m1, strict=False):
        # This if-statement can be dropped as m1 and m1 are masked.
        # But then we must zero some values: h[0, 0] = 0
        if a and b:
            h[a][id_map[b]] += 1

    # TODO - test this...
    # Requires a speed test for the export over many time frames.
    # m1 = map_array(m1, np.arange(len(map)), map)
    # h2 = _h5(m2, m1)
    # h2[0, 0] = 0
    # print("h same:", h.shape, h2.shape, h == h2)
    # print(h)
    # print(h2)

    # Greedy assignment of overlaps based on intersect over size.
    # Count size of labels.
    h1 = np.bincount(im1.reshape(-1))
    h2 = np.bincount(im2.reshape(-1))
    # Convert overlaps to a list
    overlaps = []
    # i=im1 mapped value; j=im2 value
    for j, a in enumerate(h):
        for i, c in enumerate(a):
            if c:
                # i=im1 value
                i = reverse_map[i]
                # Compute max intersect over size
                f = c / max(h1[i], h2[j])
                overlaps.append((i, j, c, f))
    overlaps.sort(reverse=True, key=lambda x: x[-1])

    # Renumber the labels.
    # Initialise as mapping to themselves.
    # Note: We use the maximum ID in the current image to offset the new image.
    omap1 = np.arange(len(h1))
    omap2 = np.arange(len(h2))
    map1 = np.zeros(len(h1), dtype=np.uint16)
    map2 = np.zeros(len(h2), dtype=np.uint16)
    m1 = len(h1)

    # TODO: Map images. Remove all overlap pixels from image 2. Re-label using maps.

    # List of overlap pixels to remove from each image
    remove1 = []
    remove2 = []

    # Remap the labels to use the ID from the object it overlaps.
    for i, j, c, _f in overlaps:
        f1 = c / h1[i]
        f2 = c / h2[j]
        # Either image could be the parent so make the largest overlap the child.
        # Assign the child to the parent ID. If the child if already assigned
        # it can be assumed that this is a smaller overlap of the child with some
        # other object. Remove the overlap child pixels.
        # If the parent is already assigned then assume a better child has already
        # overlapped the parent. Remove the overlap child pixels.
        # This works for a greedy algorithm.
        if f1 > f2:
            # current image is the child
            if map1[i]:
                remove1.append(i)
                continue  # Already assigned
            if map2[j]:
                remove1.append(i)
                continue  # Already assigned
            map2[j] = j + m1
            map1[i] = map2[j]
        else:
            # new image is the child
            if map2[j]:
                remove2.append(j)
                continue  # Already assigned
            if map1[i]:
                remove2.append(j)
                continue  # Already assigned
            map1[i] = i
            map2[j] = map1[i]

    # Remove overlaps
    if remove2:
        for v in remove2:
            im2[(im2 == v) & overlap] = 0
    if remove1:
        for v in remove1:
            im1a[(im1a == v) & overlap] = 0
        im1[yp : yp + s[0], xp : xp + s[1]] = im1a

    # Remap the new image to unique IDs (if not mapped)
    map1 = np.where(map1 == 0, omap1, map1)
    map2 = np.where(map2 == 0, omap2 + m1, map2)
    map2[0] = 0

    # Compress IDs to ascending from 1
    u = set(map1)
    u.update(map2)
    u.add(0)  # Ensure zero is added so first mapped ID is 1
    m = np.zeros(max(u) + 1, dtype=np.uint16)
    for i, v in enumerate(sorted(u)):
        m[v] = i
    for i, v in enumerate(map1):
        map1[i] = m[v]
    for i, v in enumerate(map2):
        map2[i] = m[v]

    # Remap the images
    map_array(im1, omap1, map1, out=im1)
    map_array(im2, omap2, map2, out=im2)

    # Add the remapped labels using a binary OR. Overlapping pixels have been handled
    # to match one of the parent IDs (or removed).
    im1[yp : yp + s[0], xp : xp + s[1]] |= im2

    return im1


def _h5(m1: npt.NDArray[Any], m2: npt.NDArray[Any]) -> npt.NDArray[Any]:
    m = np.max(m2) + 1
    # Multiplication must not overflow
    c = np.bincount(m1.astype(np.int_) * m + m2)
    n = math.ceil(len(c) / m)
    h = np.zeros((n, m), dtype=np.uint32)
    h.reshape(-1)[: len(c)] = c
    return h


def _merge_nonoverlapping_labels(
    im1: npt.NDArray[Any],
    im2: npt.NDArray[Any],
    xp: int = 0,
    yp: int = 0,
    m1: int = 0,
) -> npt.NDArray[Any]:
    """Merges the labels in image 2 into image 1.

    Image 2 may be smaller than image 1.

    Args:
        im1: Current labels.
        im2: New labels.
        xp: Offset in x.
        yp: Offset in y.
        m1: Maximum label in current.

    Returns:
        updated (np.array): The updated labels.
    """
    s = im2.shape
    if not m1:
        m1 = np.max(im1)
    # Remap to unique IDs.
    # Simply add the previous max to the IDs and update the max.
    # This does not compress IDs as it is assumed both inputs
    # have ascending IDs from 1.
    np.add(im2, m1, where=im2 != 0, out=im2)
    im1[yp : yp + s[0], xp : xp + s[1]] += im2

    return im1
