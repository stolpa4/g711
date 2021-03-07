G711 A-Law / u-Law audio-coding library
=======================================

The library is written as an extension for python3 and allows to work simply with G711 encoded audio.

# Requirements
1. Python 3.5+
2. Numpy (will be installed anyway before the building procedure)

# Implementation details
The library is written in pure C and relies exclusively on a python build system. I tried to minimize the use of any platform-dependend functions to provide a truly cross-platform and fast library.

# Installation
Source distribution is uploaded to PyPI, so just invoke `pip install g711`. Equivalently, you can clone code from https://github.com/stolpa4/g711 and then run `pip install <path to cloned repo>`.

# Functions
- `load_alaw(path: Union[str, os.PathLike]) -> numpy.ndarray(dtype=numpy.float32)` - Opens a file, loads its contents and decodes it from A-Law to float32 numpy array.
- `load_ulaw(path: Union[str, os.PathLike]) -> numpy.ndarray(dtype=numpy.float32)` - Same for u-Law.
- `save_alaw(path: Union[str, os.PathLike], audio_data: numpy.ndarray(dtype=numpy.float32)) -> bool` - Encodes an array to A-Law and writes bytes to a specified file. audio_data can be anything convertible to numpy.ndarray(dtype=numpy.float32).
- `save_ulaw(path: Union[str, os.PathLike], audio_data: numpy.ndarray(dtype=numpy.float32)) -> bool` - Same for u-Law.
- `decode_alaw(encoded_bts: bytes) -> numpy.ndarray(dtype=numpy.float32)` - Decodes raw A-Law bytes to float32 audio.
- `decode_ulaw(encoded_bts: bytes) -> numpy.ndarray(dtype=numpy.float32)` - Same for u-Law.
- `encode_alaw(audio_data: numpy.ndarray(dtype=numpy.float32)) -> bytes` - Encodes an array to A-Law and returns bytes object. audio_data can be anything convertible to numpy.ndarray(dtype=numpy.float32).
- `encode_ulaw(audio_data: numpy.ndarray(dtype=numpy.float32)) -> bytes` - Same for u-Law.
