from setuptools import setup, find_packages, Extension
import numpy as np
import os.path as osp

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True


def readme():
    with open('README.rst') as f:
        return f.read()


ext = '.pyx' if USE_CYTHON else '.c'


extensions = [Extension('iucm._dist',
                        sources=[osp.join("iucm", "_dist" + ext)])]


if USE_CYTHON:
    extensions = cythonize(extensions)


# read the __version__ from version.py
with open(osp.join('iucm', 'version.py')) as f:
    exec(f.read())


setup(
    name='iucm',
    version=__version__,
    description='Integrated Urban Complexity Model',
    long_description=readme(),
    ext_modules=extensions,
    packages=find_packages(exclude=['docs', 'tests*', 'examples']),
    include_dirs=[np.get_include()],
    install_requires=[
        'xarray',
        'scipy',
        'netCDF4',
        'psyplot',
        'model_organization',
        ],
    entry_points={'console_scripts': ['iucm=iucm.main:main']},
    zip_safe=False,
    data_files=[("", ["LICENSE"])],
    url='https://github.com/Chilipp/iucm',
    author='Philipp Sommer and Roger Cremades',
    author_email='philipp.sommer@unil.ch',
    )
