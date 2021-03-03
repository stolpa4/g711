from distutils.core import setup, Extension


def main():
    setup(name="g711",
          version="1.0.0",
          description="A small library load and save A-Law encoded audio files.",
          author="stolpa4",
          author_email="stolpa413@gmail.com",
          ext_modules=[Extension("g711", ["src/g711/encoder.c", "src/g711/decoder.c", 
                                          "src/g711/encoding_helpers.c", "src/g711/decoding_helpers.c",
                                          "src/g711/utils.c", "src/g711module_loader.c", "src/g711module.c"])])


if __name__ == "__main__":
    main()
