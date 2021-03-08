from pathlib import Path
from setuptools import setup, Extension

import numpy as np


def main(extension):
    setup(ext_modules=[extension],
          install_requires=['numpy>=1.20.1'],
          python_requires='>=3.6')


if __name__ == "__main__":
    source_files = [str(p) for p in Path('src').rglob('*.c')]
    extension = Extension('g711', source_files, include_dirs=[np.get_include()])
    main(extension)
