"""
Setup script for Fake Review Detection & FakeGuard Package.
==========================================================
Unified platform combining production full-stack fake review classification
with a multi-view deep learning research framework (Text, Behavior, Relational GAT).

Contributors:
- Vinay Rautulwar (Lead Developer, End-to-End System, DeBERTa & Hybrid Pipelines)
- Pramit Ghosh (Collaborator, FakeGuard Multi-View Framework & Cross-Corpus Audits)
"""

from pathlib import Path
from setuptools import find_packages, setup

readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="fake-review-detection",
    version="2.0.0",
    description="Unified Fake Review Detection Platform: Production Service & Multi-View Deep Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Vinay Rautulwar & Pramit Ghosh",
    author_email="vinayrautulwar@gmail.com",
    url="https://github.com/Vinay-rautulwar/fake-review-detection",
    package_dir={"": "."},
    packages=find_packages(include=["src", "src.*", "backend", "backend.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "joblib>=1.3.0",
        "lightgbm>=4.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "fastapi>=0.110.0",
        "pydantic>=2.6.0",
        "pydantic-settings>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "httpx>=0.27.0",
            "mongomock>=4.1.0",
            "jupyter>=1.0.0",
        ],
        "explain": [
            "shap>=0.42.0",
        ],
        "graph": [
            "torch-geometric>=2.3.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
)
