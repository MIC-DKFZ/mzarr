# MZZ Library

Mzz (Multi-Resolution Zipped Zarr) is a Python library for working with the MZZ image format, designed specifically for 2D and 3D data. MZZ provides a comprehensive solution for storing, compressing, and efficiently manipulating image data with multi-resolution views. This readme provides an overview of the MZZ library and its key features.

## Features

- **Multi-Resolution Views**: MZZ supports multi-resolution representations of image data, allowing fast and efficient viewing at different levels of detail. This feature enables quick navigation and exploration of large images.

- **Automatic Chunking**: MZZ incorporates automatic chunking for optimal performance. The library intelligently divides the image data into smaller chunks, resulting in fast loading, saving, and manipulation operations. Chunking also enables memory mapping of the image for efficient usage of system resources.

- **Lossless Compression**: MZZ utilizes the modern lossless JpegXL compressor for compressing the image data on a chunk basis. This compression method offers improved compression benefits compared to other 2D and 3D image formats. The chunk-based compression enables memory mapping of the image while still achieving high compression ratios.

- **Lossy Compression (Optional)**: In addition to lossless compression, MZZ provides an option for using lossy JpegXL compression. This option offers even better compression ratios, which can be advantageous in scenarios where the preservation of every detail is not critical.

- **Arbitrary Metadata**: MZZ supports the storage of arbitrary metadata along with the image data. This feature allows you to associate additional information, such as annotations, tags, or custom properties, with the image.

## Installation

To use the MZZ library, follow these steps:

1. Ensure that Python 3.6 or later is installed on your system.
2. Install the required dependencies by running the following command:
```
pip install numpy zarr scikit-image imagecodecs
```
3. Download the MZZ library from the official repository: [MZZ Library](https://github.com/example/mzz-library).
4. Extract the library files to your desired location.

## Getting Started

To begin using the MZZ library, you can refer to the example code provided below:

```python
import numpy as np
from mzz import Mzz

# Create or load your image array using NumPy
image = np.random.random((512, 512))

# Initialize an instance of Mzz with your image array
mzz = Mzz(array=image)

# Save the Mzz instance to disk
mzz.save(path="path/to/save.mzz")

# Load an Mzz instance from disk
loaded_mzz = Mzz(path="path/to/save.mzz")

# Access the image data as a NumPy array
loaded_image = loaded_mzz.numpy()

# Retrieve metadata associated with the Mzz instance
metadata = loaded_mzz.attrs()

# Perform operations on the image or metadata as needed
# ...
```

For more detailed usage instructions and a complete list of available methods and parameters, please refer to the MZZ library documentation.

## Contributing

We welcome contributions to the MZZ library! If you encounter any issues, have suggestions for improvements, or would like to add new features, please submit a pull request or open an issue on the official repository.

## License

The MZZ library is released under the MIT License. Feel free to use and modify the library according to your needs.


## Acknowledgments

The development of the MZZ library is made possible by the contributions of the open-source community. We would like to express our gratitude to all the individuals and organizations that have contributed to the project.

If you have any questions or need further assistance, please don't hesitate to reach out to the MZZ library maintainers or consult the documentation for additional information.