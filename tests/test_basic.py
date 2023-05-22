import unittest
from mzarr.mzarr import Mzarr
import numpy as np
import os


class TestBasic(unittest.TestCase):
    def test_slicing(self):
        array = np.ones((10, 10))
        mzz = Mzarr(array)
        mzz[0] = 2
        mzz[:, 4:7] = 3

    def test_reference(self):
        array = np.ones((10, 10))
        mzz = Mzarr(array)
        mzz[0] = 2
        np.testing.assert_array_equal(array, mzz.numpy())
        mzz[:, 4:7] = 3
        np.testing.assert_array_equal(array, mzz.numpy())

    def test_shape(self):
        array = np.ones((10, 10))
        mzz = Mzarr(array)
        assert mzz.shape == (10, 10)

    def test_save_load(self):
        array = np.ones((10, 10), dtype=np.float32)
        mzz1 = Mzarr(array)
        mzz1[0, 0] = 5.4333
        mzz1.save("tmp.mzarr")

        mzz2 = Mzarr("tmp.mzarr")
        np.testing.assert_array_equal(array, mzz2.numpy())
        mzz2.close()
        os.remove("tmp.mzarr")


if __name__ == '__main__':
    unittest.main()