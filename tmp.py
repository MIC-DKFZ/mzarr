from mzarr.mzarr import Mzarr
import numpy as np
import os

array = np.ones((10, 10), dtype=np.float32)
mzz1 = Mzarr(array)
mzz1[0, 0] = 5.4333
mzz1.save("tmp.mzarr")
mzz1.close()
# os.remove("tmp.mzarr")
mzz1[0, 1] = 7.4333
mzz1.save("tmp.mzarr")

mzz2 = Mzarr("tmp.mzarr")
print(mzz2)
# # mzz2.store.close()
mzz2.close()
os.remove("tmp.mzarr")

# import zarr
#
# # Open the Zarr array
# array = zarr.open('array.zarr', shape=(10, 10), mode='a')
# array[...] = np.ones((10, 10))
#
# # Perform operations on the array...
#
# # Close the array
# array.close()
