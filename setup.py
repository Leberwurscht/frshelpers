import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read().strip()

setuptools.setup(
    name="frseval",
    version=version,
    author="Leberwurscht",
    author_email="leberwurscht@hoegners.de",
    description="FRS helpers library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Leberwurscht/frshelpers",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy>=1.4.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3'
)
