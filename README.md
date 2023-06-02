# Mzarr

[![License Apache Software License 2.0](https://img.shields.io/pypi/l/mzarr.svg?color=green)](https://github.com/Karol-G/mzarr/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mzarr.svg?color=green)](https://pypi.org/project/mzarr)
[![Python Version](https://img.shields.io/pypi/pyversions/mzarr.svg?color=green)](https://python.org)
[![tests](https://github.com/Karol-G/mzarr/workflows/tests/badge.svg)](https://github.com/Karol-G/mzarr/actions)
[![codecov](https://codecov.io/gh/Karol-G/mzarr/branch/main/graph/badge.svg)](https://codecov.io/gh/Karol-G/mzarr)

Mzarr (Multi-Resolution Zarr) is a Python library for working with the Mzarr image format, designed specifically for 2D and 3D data. Mzarr provides a comprehensive solution for storing, compressing, and efficiently manipulating image data with multi-resolution views. This readme provides an overview of the Mzarr library and its key features.

## Features

- **Multi-Resolution Views**: Mzarr supports multi-resolution representations of image data, allowing fast and efficient viewing at different levels of detail. This feature enables quick navigation and exploration of large images. Typically, this creates only a ~10% overhead.

  **Low Memory Consumption** Mzarr requires almost no memory even when loading and vieweing large multi-dimensional images due to memory-mapping, chunking and its multi-resolution views.

- **Automatic Chunking**: Mzarr incorporates automatic chunking for optimal performance. The library intelligently divides the image data into smaller chunks, resulting in fast loading, saving, and manipulation operations. Chunking also enables memory mapping of the image for efficient usage of system resources.

- **Array-Like Slicing**: Mzarr supports array-like slicing, allowing users to extract specific regions or subsets of the image data. This feature enables the selection of portions of the image based on desired coordinates, indices, or ranges. Array-like slicing provides flexibility in manipulating and analyzing the image data by operating only on the selected regions of interest. It avoids the need to load the entire dataset into memory, resulting in efficient memory usage and faster processing times.

- **Lossless Compression**: Mzarr utilizes the modern lossless JpegXL compressor for compressing the image data on a chunk basis. This compression method offers improved compression benefits compared to other 2D and 3D image formats. The chunk-based compression enables memory mapping of the image while still achieving high compression ratios.

- **Lossy Compression (Optional)**: In addition to lossless compression, Mzarr provides an option for using lossy JpegXL compression. This option offers even better compression ratios, which can be advantageous in scenarios where the preservation of every detail is not critical.

- **Arbitrary Metadata**: Mzarr supports the storage of arbitrary metadata along with the image data. This feature allows you to associate additional information, such as annotations, tags, or custom properties, with the image.

## Installation

You can install `mzarr` via [pip](https://pypi.org/project/mzarr/):

    pip install mzarr

## Getting Started

To begin using the Mzarr library, you can refer to the example code provided below:

```python
import numpy as np
from mzarr import Mzarr

# Create or load your image array using NumPy
image = np.random.random((512, 512))

# Initialize an instance of Mzarr with your image array
mzarr = Mzarr(image)

# Manipulate the image array using slicing
mzarr[100:200, 100:200] = 1.0  # Set a 100x100 region to a value of 1.0

# Save the Mzarr instance to disk
mzarr.save(path="path/to/save.mzarr")

# Initialize an instance of Mzarr from the saved file
loaded_mzarr = Mzarr("path/to/save.mzarr")

# Access the image data as a NumPy array
loaded_image = loaded_mzarr.numpy()

# Retrieve metadata associated with the Mzarr file
metadata = loaded_mzarr.attrs()

# Perform operations on the image or metadata as needed
# ...

```

For more detailed usage instructions and a complete list of available methods and parameters, please refer to the Mzarr library documentation.

## Contributing

We welcome contributions to the Mzarr library! If you encounter any issues, have suggestions for improvements, or would like to add new features, please submit a pull request or open an issue on the official repository.

## License

The Mzarr library is released under the MIT License. Feel free to use and modify the library according to your needs.


## Acknowledgments

The development of the Mzarr library is made possible by the contributions of the open-source community. We would like to express our gratitude to all the individuals and organizations that have contributed to the project.

If you have any questions or need further assistance, please don't hesitate to reach out to the Mzarr library maintainers or consult the documentation for additional information.