from mzz import Mzz
import numpy as np
import tifffile
from numcodecs import Blosc, Zstd


# compressor = Zstd(level=15)
# image = tifffile.imread("/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/raw_data/2D3D_T1_C1_106_160kv_ct55.tif")
# image = Mzz(image)
# image.save("/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/raw_data/tmp.mzz", compressor=None)

image2 = Mzz(path="/home/k539i/Documents/datasets/original/HMGU_2022_DIADEM/dataset_WSI_mzz/train/Glucagon/Glucagon-CA0120-2021-03-2519-47-21_0000.mzz")
print(image2)
image2 = image2.numpy()
print(image2)
print(image2.sum())