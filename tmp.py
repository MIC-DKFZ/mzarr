import numpy as np
from skimage.data import binary_blobs
import zarr
import numcodecs
from imagecodecs.numcodecs import JpegXl
import shutil
import tifffile
from mzz import Mzz
from os.path import join
from natsort import natsorted
import os
import SimpleITK as sitk
import time


def load_nifti(filename, return_meta=False, is_seg=False):
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


# array1 = binary_blobs(length=1024, blob_size_fraction=0.2, n_dim=3).astype(np.uint8)
# array1 = tifffile.imread("/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/raw_data/2D3D_T1_C1_106_160kv_ct55.tif")
array1 = load_nifti("/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/raw_data/images/test/2D3D_T1_C1_160kv_15ct.nii.gz")
print(array1.dtype)
print("Min: {}, max: {}".format(np.min(array1), np.max(array1)))

# numcodecs.register_codec(JpegXl)
# # chunks = (None, None, 4)
# chunks = (512, 512, 512)
# array_zarr1 = zarr.open("tmp.zarr", shape=array1.shape, dtype=array1.dtype, mode='w', chunks=chunks, compressor=JpegXl(lossless=True))
# array_zarr1[...] = array1
#
# array_zarr2 = zarr.open("tmp.zarr", mode='r')
# print(array_zarr2.chunks)
# array2 = np.array(array_zarr2)
# print(array_zarr2.shape)
# tifffile.imwrite("tmp.tif", array2)
# print(array2)
# shutil.rmtree("tmp.zarr")

array2 = Mzz(array1)
start_time = time.time()
array2.save("/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/raw_data/2D3D_T1_C1_160kv_15ct.mzz")
print("Time: ", time.time() - start_time)

# array3 = Mzz(path="/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/raw_data/2D3D_T1_C1_160kv_15ct.mzz")
# # print(array2[512:600, 512:600, 512:600])
# array3 = array3.numpy()
# print("Min: {}, max: {}".format(np.min(array3), np.max(array3)))
# print(array3.dtype)
# print(np.array_equal(array1, array3))

import mzz

# image = mzz.Mzz(path="/home/k539i/Documents/datasets/original/HMGU_2022_DIADEM/dataset_WSI_mzz/train/Glucagon/Glucagon-CA0120-2021-03-2519-47-21_0000.mzz")
# patch = image[2000:3512, 2000:3512]
# print(patch.shape)
# print("")

# array = np.random.random(size=(500, 500))
# array_zarr = zarr.open("tmp.zarr", shape=array.shape, mode='w')
# array_zarr[...] = array
#
# array_zarr2 = zarr.open("tmp.zarr", mode='r')
# patch_zarr = array_zarr2[50:100, 50:100]
# print("")
