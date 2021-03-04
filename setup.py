from pathlib import Path
from setuptools import setup, Extension

import numpy as np


def main(extension):
    setup(name="g711",
          version="1.0.0",
          description="A small library load and save A-Law encoded audio files.",
          author="stolpa4",
          author_email="stolpa413@gmail.com",
          ext_modules=[extension])


if __name__ == "__main__":
    source_files = [str(p) for p in Path('src').rglob('*.c')]
    extension = Extension('g711', source_files, include_dirs=[np.get_include()])
    main(extension)
