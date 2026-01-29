"""
Setup script for Benchmarking QPanda3 project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="benchmarking-qpanda3",
    version="1.0.0",
    author="Syrym Zhakypbekov, Artem A. Bykov, Nurkamila A. Daurenbayeva, Kateryna V. Kolesnikova",
    author_email="s.zhakypbekov@iitu.edu.kz",
    description="Benchmarking QPanda3: A High-Performance Chinese Quantum Computing Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Syrym-Zhakypbekov/Benchmarking-QPanda3",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
)
