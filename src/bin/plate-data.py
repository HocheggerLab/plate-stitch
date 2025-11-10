#!/usr/bin/env python3
"""Program to list available plate data."""

import argparse

from plate_stitch.utils import dir_path


def main() -> None:
    """Program to list available plate data."""
    parser = argparse.ArgumentParser(
        description="""Program to list available plate data"""
    )
    parser.add_argument(
        "data", type=dir_path, nargs="+", help="Plate data directory"
    )
    args = parser.parse_args()

    # Delay imports until argument parsing succeeds
    import logging

    from plate_stitch.data import PlateData, plate_pos

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s", level=logging.INFO
    )

    for dirname in args.data:
        logger.info(dirname)
        plate = PlateData(dirname)
        out = []
        for title, values in zip(
            [
                "Wells",
                "Fields",
                "Planes",
                "Channels",
                "Mask Ch",
                "Times",
                "States",
                "FlimIDs",
            ],
            [
                plate.well_positions,
                plate.fields,
                plate.planes,
                plate.channels,
                plate.mask_channels,
                plate.times,
                plate.states,
                plate.flims,
            ],
            strict=True,
        ):
            v: list[str | int] = values  # type: ignore[assignment]
            txt = ", ".join([str(x) for x in v])
            out.append(f"{title:9s}: {txt}")
        # Load an example image to get the size and type
        row, col = plate_pos(plate.well_positions[0])
        im = plate.get_plane(
            row,
            col,
            plate.fields[0],
            plate.times[0],
            plate.channels[0],
            plate.planes[0],
        )
        out.append(f"{'Image':9s}: {im.shape} {im.dtype}")
        print("\n".join(out))


if __name__ == "__main__":
    main()
