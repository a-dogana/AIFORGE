from setuptools import setup, find_packages

setup(
    name="aiforge",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "pydantic",
        "pyyaml",
        "numpy",
        "scikit-learn",
        "pandas",
        "optuna",
        "onnx",
        "onnxruntime",
        "fastapi",
        "uvicorn",
        "httpx",
        "transformers",
        "peft",
        "datasets",
        "psutil"
    ],
    entry_points={
        "console_scripts": [
            "aiforge=aiforge_cli:main",
        ],
    },
    py_modules=["aiforge_cli"]
)
