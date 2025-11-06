# Plate Stitch

Label and stitch Operetta plate images.

## Installation

```bash
# Install uv
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone the repository
git clone https://github.com/aherbert/plate-stitch.git
# Change into the project directory
cd plate-stitch
# Create and activate virtual environment
uv sync
source .venv/bin/activate
```

## Usage

Scripts will process the exported images directory from an Operetta plate experiment.
It is assumed the image filenames match the following pattern:

    rNcNfNpN-chNskNfkNflN

where: N is a number; r=row; c=column; f=field; p=plane (Z); ch=channel; sk=timepoint;
fk=state; fl=Flim ID. Numbers prefixed by a single character are left padded with zeros
to 2 digits wide, for example `r01c01f01p01-ch1sk1fk1fl1.tiff`.

The utility script `plate-data.py` can be run on a plate data directory to provide
information about the plate: number of wells; fields; time points; planes; and channels.

The analysis pipeline will:

1. Generate a flat-field correction image for each channel. This image is generated
by sampling fields and timepoints from the plate; summing the image samples; smoothing
the combined image; and normalising to a mean of 1. Dividing by this flat-field
image should level the intensity across the field of the image accounting for drop-off
at the camera edges. The image is saved to the plate data directory as `flatfield.tiff`.

1. Segment the nuclei channel of the image. Segmentation masks are saved to the
plate data directory using the same image naming convention with a `-mask.tiff` suffix.
Segmentation uses `cellpose` which requires that the named model be installed in the
`cellpose` models directory. This can be achieved using:

        cellpose --add_model [model path]

1. Stitch the well images and export to a named directory. The fields of the plate
well are stiched using known patterns used by the Operetta microscope, e.g. 5x5; 3x3.
Stitching is performed on the images using pixel blending, and on the segmentation masks
using overlap analysis to maintain object identities.
The exported images are placed in a sub-directory for each well and named
`iT.tiff` and `mT.tiff` for the image and mask respectively, where `T` is
the time point.

The pipeline steps above can be run individually using scripts, or combined into a
single analysis pipeline with a master script.

## Development

This project uses [pre-commit](https://pre-commit.com/) to create actions to validate
changes to the source code for each `git commit`.
Install the hooks for your development repository clone using:

    pre-commit install

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
