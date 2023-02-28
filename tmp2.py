import mzz
import numpy as np

seg = mzz.Mzz()
seg.load("/home/k539i/Documents/network_drives/cluster-data/original/HMGU_2022_DIADEM/predictions/WSI/nnunetv2/Dataset107_DIADEMv3/train/Insulin/Insulin-CA0120-2021-03-1817-51-58.mzz")
seg = seg.numpy().astype(np.uint8)
np.save("/home/k539i/Documents/datasets/original/HMGU_2022_DIADEM/Insulin-CA0120-2021-03-1817-51-58.npy", seg)