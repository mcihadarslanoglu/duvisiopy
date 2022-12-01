import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="duvisiopy",
    version="0.0.1",
    author="mcihadarslanoglu",
    author_email="cihatdt.21@gmail.com",
    description="This library extend functionalty of any pytorch model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mcihadarslanoglu/duvisiopy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
