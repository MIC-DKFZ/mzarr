import argparse
from os.path import join
import os
from os.path import join
from natsort import natsorted
import tifffile
from mzarr import Mzarr
import numpy as np


def tiff2mzarr(load_dir: str, save_dir: str, is_seg: bool, lossy: bool) -> None:
    """
    Converts a tiff files of an image to a mzarr file.

    Args:
        load_dir (str): Path to the tiff directory.
        save_dir (str): Path to the folder where the mzarr file should be saved.
        is_seg (bool): Whether the image is a segmentation.
        lossy (bool): Whether lossy JpegXL compression should be used.
    """
    name = os.path.basename(os.path.normpath(load_dir))
    filepaths = load_filepaths(load_dir)
    image = tifffile.imread(filepaths)
    image = Mzarr(image)    
    image.save(join(save_dir, "{}.mzarr".format(name)), is_seg=is_seg, lossless=not lossy)


def load_filepaths(load_dir: str, extension: str = None, return_path: bool = True, return_extension: bool = True) -> np.ndarray:
    """
    Given a directory path, returns an array of file paths with the specified extension.

    Args:
        load_dir: The directory containing the files.
        extension: A string or list of strings specifying the file extension(s) to search for. Optional.
        return_path: If True, file paths will include the directory path. Optional.
        return_extension: If True, file paths will include the file extension. Optional.

    Returns:
        An array of file paths.
    """
    filepaths = []
    if isinstance(extension, str):
        extension = tuple([extension])
    elif isinstance(extension, list):
        extension = tuple(extension)
    elif extension is not None and not isinstance(extension, tuple):
        raise RuntimeError("Unknown type for argument extension.")

    if extension is not None:
        extension = list(extension)
        for i in range(len(extension)):
            if extension[i][0] != ".":
                extension[i] = "." + extension[i]
        extension = tuple(extension)

    for filename in os.listdir(load_dir):
        if extension is None or str(filename).endswith(extension):
            if not return_extension:
                if extension is None:
                    filename = filename.split(".")[0]
                else:
                    for ext in extension:
                        if str(filename).endswith((ext)):
                            filename = str(filename)[:-len(ext)]
            if return_path:
                filename = join(load_dir, filename)
            filepaths.append(filename)
    filepaths = np.asarray(filepaths)
    filepaths = natsorted(filepaths)

    return filepaths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the directory containing the TIFF files.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the folder that should be used for the Mzarr images.")
    parser.add_argument('--seg', required=False, default=False, action="store_true", help="Whether the image is a segmentation.")
    parser.add_argument('--lossy', required=False, default=False, action="store_true", help="Whether lossy JpegXL compression should be used.")
    args = parser.parse_args()

    tiff2mzarr(args.input, args.output, args.seg, args.lossy)
