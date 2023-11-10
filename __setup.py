from setuptools import setup

setup(
    name="BIDS",
    version="0.0.2",
    author="Robert Graf",
    author_email="robert.graf@tum.de",
    packages=["BIDS", "BIDS.test"],
    # scripts=["bin/script1", "bin/script2"],
    # url="http://pypi.python.org/pypi/PackageName/",
    license="LICENSE.txt",
    description="A collection of tools, that work with files in a (weak) BIDS standard",
    long_description=open("README.md").read(),
    install_requires=[
        "pathlib",
        "pytest",
        "nibabel",
        "numpy",
        "antspyx",
        "typing_extensions",
        "scipy",
        "dataclasses",
        "SimpleITK",
        "matplotlib",
        "dicom2nifti",
        "func_timeout",
        "dill",
    ],
)

# Build from source:
# python setup.py build
# And install:
# python setup.py install
# Under Development
# Develop mode is really, really nice:
# $ python setup.py develop
# sudo python3 setup.py develop
# or:
# $ pip install -e ./

# which python
#
# sudo /home/robert/anaconda3/envs/py3.10/bin/python setup.py develop
