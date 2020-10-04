import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="module-classifier-api",
    version="0.0.1",
    author="Carsten Schnober",
    author_email="carschno@gmail.com",
    description="API for The Syllabus modules classifier.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    # conda install -c conda-forge fasttext
    install_requires=["fasttext==0.9.2"],
    test_requires=["pytest-subtests==0.3.2", "pytest-mock==3.3.1"],
)
