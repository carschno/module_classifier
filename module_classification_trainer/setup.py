import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="module-trainer",
    version="0.0.5",
    author="Carsten Schnober",
    author_email="carschno@gmail.com",
    description="Train a module classification model for The Syllabus.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where="src", exclude=("tests",)),
    package_dir={"": str("src")},
    scripts=["src/train_module_classifier"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=["fasttext==0.9.2", "module-classifier-api==0.0.5"],
    extras_require={"testing": ["pytest==6.1.0", "pytest-mock==3.3.1"]},
)
