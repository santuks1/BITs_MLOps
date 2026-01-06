from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="heart-disease-prediction",
    version="1.0.0",
    author="MLOps Engineer",
    description="ML model for predicting heart disease risk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mlops-heart-disease",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "mlflow>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pylint>=2.0",
            "black>=22.0",
            "jupyter>=1.0",
        ],
        "monitoring": [
            "prometheus-client>=0.15.0",
        ],
    },
)