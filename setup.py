from setuptools import setup, find_packages

setup(
    name="dyra-iiot",
    version="1.0.0",
    description=(
        "DyRA-IIoT: A Hybrid Framework for Asset-Aware Dynamic Risk "
        "Assessment in IIoT Networks"
    ),
    author="Akylbek Tokhmetov, Mansiya Kantureyeva, Liliya Tanchenko, "
           "Ainagul Alimagambetova",
    author_email="tokhmetov_ab@enu.kz",
    url="https://github.com/enu-cybersec/DyRA-IIoT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "onnx>=1.15.0",
        "onnxruntime>=1.16.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
)
