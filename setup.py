from setuptools import setup, find_packages

setup(
    name="aiforge",
    version="1.2.0rc1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "pydantic",
        "pyyaml",
        "numpy",
        "scikit-learn",
        "pandas",
    ],
    extras_require={
        "tuning": ["optuna"],
        "export": ["onnx", "onnxruntime"],
        "serving": ["fastapi", "uvicorn", "httpx"],
        "llm": ["transformers", "peft", "datasets", "psutil", "accelerate"],
        "all": [
            "optuna", "onnx", "onnxruntime", "fastapi", "uvicorn", 
            "httpx", "transformers", "peft", "datasets", "psutil", "accelerate"
        ]
    },
    entry_points={
        "console_scripts": [
            "aiforge=aiforge_cli:main",
        ],
    },
    py_modules=["aiforge_cli"]
)
