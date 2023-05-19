import numpy as np
import zarr
from zarr.util import guess_chunks, normalize_dtype
from skimage.transform import pyramid_gaussian
from imagecodecs.numcodecs import JpegXl
from typing import Optional, List


class Mzz:
    def __init__(self, array: Optional[np.ndarray] = None, path: Optional[str] = None):
        """
        Initialize an instance of Mzz.

        Args:
            array (np.ndarray, optional): The array to be stored. Defaults to None.
            path (str, optional): The path to load the array from. Defaults to None.
        """
        self.store = None
        self.array = array

        if path is not None:
            self.load(path)

        if self.array is not None:
            self.shape = self.array.shape

    def save(
            self,
            path: str,
            properties=None,
            num_pyramids: int = 4,
            channel_axis: Optional[int] = None,
            is_seg: bool = False,
            type: str = "subsampled",
            lossless: bool = True,
            chunks: bool = True
    ):
        """
        Save the Mzz instance to disk.

        Args:
            path (str): The path to save the Mzz instance to.
            properties: Additional properties to be saved.
            num_pyramids (int): The number of pyramids to create. Defaults to 4.
            channel_axis (int, optional): The axis representing channels. Defaults to None.
            is_seg (bool): Whether the array is a segmentation mask. Defaults to False.
            type (str): The type of pyramid to create ("gaussian" or "subsampled"). Defaults to "subsampled".
            lossless (bool): Whether to use lossless compression. Defaults to True.
            chunks (bool): Whether to use chunked storage. Defaults to True.
        """
        if self.store is not None:
            self.array = self.store["base"]
        pyramid = self._create_pyramid(self.array, num_pyramids, channel_axis, is_seg, type)
        self._save(path, properties, pyramid, type, is_seg, lossless, chunks, channel_axis)

    def load(self, path: str):
        """
        Load the Mzz instance from disk.

        Args:
            path (str): The path to load the Mzz instance from.
        """
        self.store = zarr.open(zarr.ZipStore(path, mode='r'), mode="r")
        self.array = self.store["base"]
        self.shape = self.array.shape

    def numpy(self) -> np.ndarray:
        """
        Get a NumPy array representation of the Mzz instance.

        Returns:
            np.ndarray: The NumPy array representation of the Mzz instance.
        """
        if self.store is not None:
            self.array = self.store["base"]
        return np.array(self.array)

    def attrs(self) -> dict:
        """
        Get the attributes of the Mzz instance.

        Returns:
            dict: The attributes of the Mzz instance.
        """
        return dict(self.store.attrs)

    def __getitem__(self, key):
        """
        Get an item from the Mzz instance.

        Args:
            key: The key to access the item.

        Returns:
            Any: The item corresponding to the key.
        """
        if self.store is not None:
            self.array = self.store["base"]
        return self.array[key]

    def _create_pyramid(self, array: np.ndarray,
                        num_pyramids: int,
                        channel_axis: Optional[int],
                        is_seg: bool,
                        type: str
                        ) -> List[np.ndarray]:
        """
        Create a pyramid from the given array.

        Args:
            array (np.ndarray): The input array.
            num_pyramids (int): The number of pyramids to create.
            channel_axis (Optional[int]): The axis representing channels.
            is_seg (bool): Whether the array is a segmentation mask.
            type (str): The type of pyramid to create ("gaussian" or "subsampled").

        Returns:
            List[np.ndarray]: The pyramid of arrays.
        Raises:
            RuntimeError: If an unknown pyramid type is specified.
        """
        if num_pyramids is None or num_pyramids == 0:
            return [array]
        if type == "gaussian":
            order = 1
            if is_seg:
                order = 0
            pyramid = list(pyramid_gaussian(array, downscale=2, max_layer=num_pyramids, channel_axis=channel_axis, order=order, preserve_range=True))
            pyramid = [p.astype(array.dtype) for p in pyramid]
        elif type == "subsampled":
            pyramid = [array]
            if channel_axis is not None and channel_axis < 0:
                channel_axis = len(array.shape) + channel_axis
            slices = []
            for axis in range(len(array.shape)):
                if channel_axis is None or axis != channel_axis:
                    slices.append(slice(None, None, 2))
                else:
                    slices.append(slice(None, None, None))
            for _ in range(num_pyramids):
                pyramid.append(pyramid[-1][tuple(slices)])
            # for _ in range(num_pyramids):
            #     sub_sampled_array = pyramid[-1]
            #     for axis in range(len(array.shape)):
            #         if channel_axis is None or axis != channel_axis:
            #             sub_sampled_array = np.delete(sub_sampled_array, np.s_[::10], axis=axis)
            #     pyramid.append(sub_sampled_array)

        else:
            raise RuntimeError("Unknown pyramid type.")

        return pyramid

    def _save(
            self,
            filepath: str,
            properties,
            pyramid: List[np.ndarray],
            pyramid_type: str,
            is_seg: bool,
            lossless: bool,
            chunks: bool,
            channel_axis: Optional[int],
    ):
        """
        Save the Mzz instance to disk.

        Args:
            filepath (str): The path to save the Mzz instance to.
            properties: Additional properties to be saved.
            pyramid (List[np.ndarray]): The pyramid of arrays to be saved.
            pyramid_type (str): The type of pyramid to create ("gaussian" or "subsampled").
            is_seg (bool): Whether the array is a segmentation mask.
            lossless (bool): Whether to use lossless compression.
            chunks (bool): Whether to use chunked storage.
            channel_axis (Optional[int]): The axis representing channels.
        """
        if (chunks is None or chunks is True) and channel_axis is not None:
            dtype = normalize_dtype(pyramid[0].dtype, None)[0]
            chunks = guess_chunks(pyramid[0].shape, dtype.itemsize)
            chunks = list(chunks)
            chunks[channel_axis] = pyramid[0].shape[channel_axis]
            chunks = tuple(chunks)
        with zarr.ZipStore(filepath, compression=0, mode='w') as store:
            grp = zarr.group(store)

            series = []
            for p, dataset in enumerate(pyramid):
                if p == 0:
                    path = "base"
                    p_lossless = lossless
                else:
                    path = "{}_{}".format(pyramid_type, p)
                    p_lossless = False
                grp.create_dataset(path, data=pyramid[p], chunks=chunks, compressor=JpegXl(lossless=p_lossless), dtype=pyramid[p].dtype)
                series.append({"path": path})

            multiscale = {
                "version": "0.1",
                "datasets": series,
                "type": pyramid_type,
            }

            grp.attrs["multiscale"] = multiscale
            grp.attrs["seg"] = is_seg
            grp.attrs['properties'] = properties
            grp.attrs["lossless"] = lossless
            grp.attrs["channel_axis"] = channel_axis
            grp.attrs["num_spatial"] = len(pyramid[0].shape) if channel_axis is None else len(pyramid[0].shape) - 1