import sys
import os.path
# https://github.com/python-versioneer/python-versioneer/issues/193
sys.path.insert(0, os.path.dirname(__file__))

from setuptools import setup, find_packages
import codecs

import versioneer



def main():

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setup(
        name="axivity-outdoor-light",
        version=versioneer.get_version(),    # Do not edit
        cmdclass=versioneer.get_cmdclass(),  # Do not edit
        python_requires=">=3.8, <4",
        description="Axivity Outdoor Light Exposure Analysis",
        long_description=long_description,  # Do not edit. See README.md
        long_description_content_type="text/markdown",
        keywords="example, setuptools, versioneer",
        url="https://github.com/OxWearables/axivity-outdoor-light",
        download_url="https://github.com/OxWearables/axivity-outdoor-light",
        author=get_string("__author__"),                      # Do not edit. see src/axivity_outdoor_light/__init__.py
        maintainer=get_string("__maintainer__"),              # Do not edit. see src/axivity_outdoor_light/__init__.py
        maintainer_email=get_string("__maintainer_email__"),  # Do not edit. See src/axivity_outdoor_light/__init__.py
        license=get_string("__license__"),                    # Do not edit. See src/axivity_outdoor_light/__init__.py

        # This is for PyPI to categorize your project. See: https://pypi.org/classifiers/
        classifiers=[
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
        ],

        # Where to find the source code
        packages=find_packages(where="src", exclude=("test", "tests")),
        package_dir={"": "src"},
        # What other files to include, e.g. *.class if the package uses some Java code.
        package_data={"axivity_outdoor_light": ["*.txt", "*.rst", "*.md"]},

        # This option will include all files in the `src/axivity_outdoor_light` directory provided they
        # are listed in the `MANIFEST.in` file, OR are being tracked by git.
        # See: https://setuptools.pypa.io/en/latest/userguide/datafiles.html
        # include_package_data=True,

        # Dependencies
        install_requires=[
            "actipy>=3.0.5",
            "numpy==1.25.*",
            "pandas==2.0.*",
            "scipy==1.11.*",
            "statsmodels==0.14.*",
            "scikit-learn==1.3.*",
            "tqdm==4.64.*",
            "matplotlib==3.8.*",
        ],

        # Optional packages. Can be installed with:
        # `$ pip install axivity-outdoor-light[dev]` or
        # `$ pip install axivity-outdoor-light[docs]` or
        # `$ pip install axivity-outdoor-light[dev,docs]`
        extras_require={
            # Will be installed with `$ pip install axivity-outdoor-light[dev]`
            "dev": [
                "versioneer",
                "twine",
                "ipdb",
                "flake8",
                "autopep8",
            ],
            # Will be installed with `$ pip install axivity-outdoor-light[docs]`
            "docs": [
                "sphinx>=4.2",
                "sphinx_rtd_theme>=1.0",
                "readthedocs-sphinx-search>=0.1",
                "docutils<0.18",
            ]
        },

        # Define entry points for command-line scripts, e.g.: `$ hello --name Alice`
        entry_points={
            "console_scripts": [
                "axlux=axivity_outdoor_light.main:main",
            ],
        },

    )


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_string(string, rel_path="src/axivity_outdoor_light/__init__.py"):
    for line in read(rel_path).splitlines():
        if line.startswith(string):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError(f"Unable to find {string}.")



if __name__ == "__main__":
    main()
