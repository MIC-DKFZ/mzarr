import argparse
from tqdm import tqdm
from os.path import join
import os
from natsort import natsorted
import SimpleITK as sitk
from mzarr import Mzarr
import numpy as np
from typing import Union, Tuple


def all_nifti2mzarr(load_dir: str, save_dir: str, is_seg: bool, lossy: bool) -> None:
    """
    Converts all nifti files into mzarr files.

    Args:
        load_dir (str): Directory where the nifti files are located.
        save_dir (str): Directory where the mzarr files should be saved.
        is_seg (bool): Whether the image is a segmentation.
        lossy (bool): Whether lossy JpegXL compression should be used.
    """
    names = load_filepaths(load_dir, extension=".nii.gz", return_path=False, return_extension=False)
    for name in tqdm(names, desc="Image conversion"):
        nifti2mzarr(join(load_dir, name + ".nii.gz"), join(save_dir, name + ".mzarr"), is_seg, lossy)


def nifti2mzarr(load_filepath: str, save_filepath: str, is_seg: bool, lossy: bool) -> None:
    """
    Converts a single nifti file to a mzarr file.

    Args:
        load_filepath (str): Path to the nifti file.
        save_dir (str): Path to where the mzarr file should be saved.
                is_seg (bool): Whether the image is a segmentation.
        lossy (bool): Whether lossy JpegXL compression should be used.
    """
    image, spacing, affine, header = load_nifti(load_filepath, return_meta=True)
    image = Mzarr(image)
    image.save(save_filepath, attrs={"spacing": spacing}, is_seg=is_seg, lossless=not lossy)


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


def load_nifti(filename: str, return_meta: bool = False, is_seg: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[float, float, float], None, dict]]:
    """
    Load a NIfTI file and return it as a numpy array.

    Args:
        filename: The path to the NIfTI file.
        return_meta: If True, return the image metadata. Optional.
        is_seg: If True, round image values to nearest integer. Optional.

    Returns:
        The NIfTI file as a numpy array. If return_meta is True, a tuple with the image numpy array, the image
        spacing, affine transformation matrix and image metadata dictionary will be returned.
    """
    image = sitk.ReadImage(filename)
    image_np = sitk.GetArrayFromImage(image)

    if is_seg:
        image_np = np.rint(image_np)
        # image_np = image_np.astype(np.int16)  # In special cases segmentations can contain negative labels, so no np.uint8

    if not return_meta:
        return image_np
    else:
        spacing = image.GetSpacing()
        keys = image.GetMetaDataKeys()
        header = {key:image.GetMetaData(key) for key in keys}
        affine = None  # How do I get the affine transform with SimpleITK? With NiBabel it is just image.affine
        return image_np, spacing, affine, header


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the file(s) that should be converted to from Nifti to Mzarr.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the folder that should be used for the Mzarr images.")
    parser.add_argument('--seg', required=False, default=False, action="store_true", help="Whether the image is a segmentation.")
    parser.add_argument('--lossy', required=False, default=False, action="store_true", help="Whether lossy JpegXL compression should be used.")
    args = parser.parse_args()

    if not args.input.endswith(".nii.gz"):
        all_nifti2mzarr(args.input, args.output, args.seg, args.lossy)
    else:
        nifti2mzarr(args.input, args.output, args.seg, args.lossy)
