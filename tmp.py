import numpy as np
from skimage.data import binary_blobs
import zarr
import numcodecs
from imagecodecs.numcodecs import JpegXl


array1 = binary_blobs(length=128, blob_size_fraction=0.2, n_dim=3).astype(np.uint8)
numcodecs.register_codec(JpegXl)
array_zarr1 = zarr.open("tmp.zarr", shape=array1.shape, dtype=np.uint8, mode='w', compressor=JpegXl(lossless=True, planar=False), chunks=(None, None, 4))
array_zarr1[...] = array1

array_zarr2 = zarr.open("tmp.zarr", mode='r')
print(array_zarr2.shape)
array2 = np.array(array_zarr2)
print(array2)

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
