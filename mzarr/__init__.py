__version__ = "0.0.3"

import numcodecs
from imagecodecs.numcodecs import JpegXl
from mzarr.mzarr import Mzarr

numcodecs.register_codec(JpegXl)
