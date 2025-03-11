from setuptools import find_packages, setup

setup(
    name="dfreproject",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.17.0",
        "astropy>=4.0",
        "torch>=1.9.0",
    ],
    extras_require={
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autoapi>=2.0.0",
            "nbsphinx>=0.8.9",
            "ipykernel>=6.0.0",
            "matplotlib>=3.4.0",
        ],
    },
    python_requires=">=3.7",
    author="Carter Rhea",
    author_email="carter.rhea@dragonfly1000.com",
    description="A package for reprojecting astronomical images",
    long_description="A package that implements astronomical image dfreproject with SIP distortion correction",
    url="https://github.com/DragonflyTelescope/dfreproject",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
