from setuptools import setup, find_namespace_packages, find_packages

setup(name='mzz',
      packages=find_namespace_packages(include=["mzz", "mzz.*"]),
      version='0.0.1',
      description='none',
      url='127.0.0.1',
      author_email='karol.gotkowski@dkfz.de',
      license='Apache License, Version 2.0',
      install_requires=[
            "numpy",
            "scikit-image>=0.19",
            "zarr",
            'numcodecs',
            'imagecodecs==2023.1.23'
      ],
      zip_safe=False)
