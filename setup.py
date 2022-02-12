import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="module_classifier",
    version="0.4.2",
    author="Carsten Schnober",
    author_email="carschno@gmail.com",
    description="API for The Syllabus modules classifier.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where="./src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=[
        "src/scripts/train_module_classifier",
        "src/scripts/explain",
        "src/scripts/train_main_edition_classifier.py",
    ],
    python_requires=">=3.8",
    # conda install -c conda-forge fasttext
    install_requires=[
        "fasttext==0.9.2",
        "pydantic>=1.8.0<1.9.0",
        "lime>=0.2.0,<0.3.0",
        "boto3>=1.20.0,<1.21.0",
    ],
    extras_require={"testing": ["pytest>=6.2.5,<6.3.0", "pytest-mock==3.3.1"]},
)
