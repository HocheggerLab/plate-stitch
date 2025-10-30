"""Module for loading plate experiment data."""


class PlateData:
    """
    Provides data loading of an Operetta plate experiment.

    The data is expected to be in the format of single plane tiff files
    named using the following convention:

        rNcNfNpN-chNskNfkNflN

    where: N is a number; r=row; c=column; f=field; p=plane (Z);
    ch=channel; sk=timepoint; fk=state; fl=Flim ID.
    """

    def __init__(self, path: str) -> None:
        """Initialises the plate experiment.

        Args:
            path: Path to the plate images directory.
        """
        # TODO - discover the plate experiment image files
        pass

    # TODO method to load an image as a numpy array TCZYX
