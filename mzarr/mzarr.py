import numpy as np
import zarr
from zarr.util import guess_chunks, normalize_dtype
from skimage.transform import pyramid_gaussian
from imagecodecs.numcodecs import JpegXl
from typing import Optional, List, Union, Literal, Any
import os


class Mzarr:
    def __init__(self, store: Union[np.ndarray, str], mode: Literal['r', 'r+', 'a', 'w', 'w-'] = 'a') -> None:
        """
        Initialize the Mzarr instance.

        Args:
            store (Union[np.ndarray, str]): The array to be handled by the instance, or the
                path to a Mzarr file to be loaded.
            mode (Literal['r', 'r+', 'a', 'w', 'w-'], optional): The mode in which to open
                the Mzarr file. This is only used if `store` is a path.
                ‘r’ means read only (must exist);
                ‘r+’ means read/write (must exist);
                ‘a’ means read/write (create if doesn’t exist);
                ‘w’ means create (overwrite if exists);
                ‘w-’ means create (fail if exists).
                Defaults to 'a'.
        """

        self.path = None
        self.store = None
        self.array = None

        if isinstance(store, str):
            self.load(store, mode)
        else:
            self.array = store

    def load(self, path: str, mode: Literal['r', 'r+', 'a', 'a'] = 'a') -> None:
        """
        Load the Mzarr instance from a file on disk.

        Args:
            path (str): The path to the Mzarr file to load.
            mode (Literal['r', 'r+', 'a', 'a'], optional): The mode in which to open the Mzarr file.
                Defaults to 'a'.
        """

        self.path = path
        self.store = zarr.open(zarr.ZipStore(path, mode=mode), mode=mode)
        self.array = self.store["base"]

    def save(
            self,
            path: str,
            attrs: Optional[dict] = None,
            num_pyramids: int = 4,
            channel_axis: Optional[int] = None,
            is_seg: bool = False,
            type: Literal['subsampled', 'gaussian'] = "subsampled",
            lossless: bool = True,
            chunks: bool = True,
            mode: Literal['r+', 'a', 'w', 'w-'] = 'a',
            overwrite: bool = True
    ) -> None:
        """
        Save the Mzarr instance to a file on disk. This includes creating a pyramid of images,
        and writing the pyramid along with metadata to disk.

        Args:
            path (str): The path to save the Mzarr file to.
            attrs (dict, optional): Additional attributes to be saved in the Mzarr file. Defaults to None.
            num_pyramids (int, optional): The number of pyramid levels to create. Defaults to 4.
            channel_axis (int, optional): The axis of the array representing channels. Defaults to None.
            is_seg (bool, optional): Whether the array represents a segmentation mask. Defaults to False.
            type (str, optional): The type of pyramid to create ("gaussian" or "subsampled"). Defaults to "subsampled".
            lossless (bool, optional): Whether to use lossless compression. Defaults to True.
            chunks (bool, optional): Whether to use chunked storage. Defaults to True.
            mode (Literal['r+', 'a', 'w', 'w-'], optional): The mode in which to open the Mzarr file. Defaults to 'a'.
            overwrite (bool, optional): Whether to overwrite an existing file at the same path. Defaults to True.
        """

        pyramid = self._create_pyramid(self.array, num_pyramids, channel_axis, is_seg, type)
        self._save(path, attrs, pyramid, type, is_seg, lossless, chunks, channel_axis, mode, overwrite)

    def numpy(self) -> np.ndarray:
        """
        Get a NumPy array representation of the Mzarr instance.

        This method returns a copy of the base array of the Mzarr instance as a NumPy array.

        Returns:
            np.ndarray: The NumPy array representation of the Mzarr instance.
        """
        return np.array(self.array)

    def close(self) -> None:
        """
        Close the Mzarr file associated with the Mzarr instance.

        This method closes the underlying ZipStore associated with the Mzarr instance.
        """

        self.store.store.close()

    def attrs(self) -> dict:
        """
        Get the attributes of the Mzarr instance.

        Returns:
            dict: The attributes of the Mzarr instance.
        """
        return dict(self.store.attrs)

    def __getitem__(self, key: Union[int, slice]) -> np.ndarray:
        """
        Get an item or a slice from the base array of the Mzarr instance.

        This method supports integer and slice indexing with the same semantics as NumPy arrays.

        Args:
            key (Union[int, slice]): The index or slice to access in the base array.

        Returns:
            np.ndarray: The item or slice from the base array corresponding to the key.
        """

        return self.array[key]

    def __setitem__(self, key: Union[int, slice], value: np.ndarray) -> None:
        """
        Set an item or a slice in the base array of the Mzarr instance.

        This method supports integer and slice indexing with the same semantics as NumPy arrays.

        Args:
            key (Union[int, slice]): The index or slice to access in the base array.
            value (np.ndarray): The value to set at the specified index or slice.
        """

        self.array.__setitem__(key, value)

    def __getattr__(self, name: str) -> Any:
        """
        Get a named attribute from the base array of the Mzarr instance.

        This method is called when an attribute lookup has not found the attribute in the usual places
        (i.e., it is not an instance attribute nor is it found in the class tree for self). name is the
        attribute name. This method should return the (computed) attribute value or raise an AttributeError exception.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The attribute value.
        """

        return getattr(self.array, name)

    def __repr__(self) -> str:
        """
        Get a string representation of the Mzarr instance.

        Returns:
            str: The string representation of the Mzarr instance.
        """

        return repr(self.array[...])

    def _create_pyramid(self,
                        array: np.ndarray,
                        num_pyramids: int,
                        channel_axis: Optional[int],
                        is_seg: bool,
                        type: Literal['subsampled', 'gaussian']
                        ) -> List[np.ndarray]:
        """
        Create a pyramid from the given array.

        This method generates a pyramid of images from the given input array. The pyramid
        can be of "gaussian" or "subsampled" type and will contain 'num_pyramids' levels.

        Args:
            array (np.ndarray): The input array.
            num_pyramids (int): The number of pyramid levels to create.
            channel_axis (Optional[int]): The axis representing channels in the array.
            is_seg (bool): Indicates if the array is a segmentation mask.
            type (str): The type of pyramid to create ("gaussian" or "subsampled").

        Returns:
            List[np.ndarray]: The pyramid of arrays.
        """

        if num_pyramids is None or num_pyramids == 0:
            return [array]
        if type == "gaussian":
            order = 1
            if is_seg:
                order = 0
            pyramid = list(pyramid_gaussian(array, downscale=2, max_layer=num_pyramids, channel_axis=channel_axis, order=order, preserve_range=True))
            pyramid = [p.astype(array.dtype) for p in pyramid]
        else:
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

        return pyramid

    def _save(self,
              path: str,
              attrs: Optional[dict],
              pyramid: List[np.ndarray],
              pyramid_type: Literal['subsampled', 'gaussian'],
              is_seg: bool,
              lossless: bool,
              chunks: bool,
              channel_axis: Optional[int],
              mode: Literal['r+', 'a', 'w', 'w-'] = 'a',
              overwrite: bool = True
              ) -> None:
        """
        Save the Mzarr instance to disk.

        This method saves the pyramid of images along with associated attributes to
        a specified location in a disk. It also manages compression and chunking options.

        Args:
            path (str): The path to save the Mzarr instance to.
            attrs (Optional[dict]): Additional attributes to be saved.
            pyramid (List[np.ndarray]): The pyramid of arrays to be saved.
            pyramid_type (str): The type of pyramid ("gaussian" or "subsampled").
            is_seg (bool): Whether the array is a segmentation mask.
            lossless (bool): Whether to use lossless compression.
            chunks (bool): Whether to use chunked storage.
            channel_axis (Optional[int]): The axis representing channels in the array.
            mode (Literal['r+', 'a', 'w', 'w-']): The mode in which to open the Mzarr file. Default is 'a'.
            overwrite (bool): Whether to overwrite an existing Mzarr file at the same path.

        Raises:
            RuntimeError: If a file already exists at the specified path and 'overwrite' is set to False.
        """

        if os.path.exists(path) and overwrite:
            os.remove(path)
        elif os.path.exists(path):
            raise RuntimeError("A file already exists under {}".format(path))

        if (chunks is None or chunks is True) and channel_axis is not None:
            dtype = normalize_dtype(pyramid[0].dtype, None)[0]
            chunks = guess_chunks(pyramid[0].shape, dtype.itemsize)
            chunks = list(chunks)
            chunks[channel_axis] = pyramid[0].shape[channel_axis]
            chunks = tuple(chunks)

        zip_store = zarr.ZipStore(path, compression=0, mode=mode)
        grp = zarr.group(zip_store)

        series = []
        for p, dataset in enumerate(pyramid):
            if p == 0:
                resolution_path = "base"
                p_lossless = lossless
            else:
                resolution_path = "{}_{}".format(pyramid_type, p)
                p_lossless = False
            grp.create_dataset(resolution_path, data=pyramid[p], chunks=chunks, compressor=JpegXl(lossless=p_lossless), dtype=pyramid[p].dtype)
            series.append({"path": resolution_path})

        multiscale = {
            "version": "0.1",
            "datasets": series,
            "type": pyramid_type,
        }

        grp.attrs["multiscale"] = multiscale
        grp.attrs["seg"] = is_seg
        grp.attrs['attrs'] = attrs
        grp.attrs["lossless"] = lossless
        grp.attrs["channel_axis"] = channel_axis
        grp.attrs["num_spatial"] = len(pyramid[0].shape) if channel_axis is None else len(pyramid[0].shape) - 1
        zip_store.close()

        self.store = grp