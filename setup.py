"""
Setup configuration for Hybrid Intelligence Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hybrid-intelligence-framework",
    version="1.0.0",
    author="Authors",
    author_email="email@example.com",
    description="Multi-scale physical constraints and data synergy framework for small-sample scientific prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/hybrid-intelligence-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "torch>=1.9.0",
        "scikit-learn>=1.0.1",
        "xgboost>=1.5.0",
        "bayesian-optimization>=1.2.0",
        "statsmodels>=0.12.0",
        "joblib>=1.1.0",
        "openpyxl>=3.0.9",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "viz": [
            "matplotlib>=3.4.3",
            "seaborn>=0.11.2",
        ]
    },
    entry_points={
        "console_scripts": [
            "hif-train=examples.train_model:main",
            "hif-predict=examples.predict_solubility:main",
        ],
    },
)