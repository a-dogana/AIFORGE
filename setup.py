from setuptools import setup, find_packages

EXTRAS = {
    "tuning": ["optuna"],
    "export": ["onnx", "onnxruntime"],
    "serving": ["fastapi", "uvicorn", "httpx"],
    "ui": ["fastapi", "uvicorn", "httpx"],
    "generation": ["openai"],
    "llm": ["transformers", "peft", "datasets", "accelerate", "openai"],
}
EXTRAS["all"] = sorted({pkg for group in EXTRAS.values() for pkg in group})

setup(
    name="aiforge",
    version="2.1.1",
    packages=find_packages(include=["aiforge", "aiforge.*"]),
    include_package_data=True,
    package_data={
        "aiforge.ui": [
            "static/*",
            "templates/*",
        ],
    },
    install_requires=[
        "torch",
        "torchvision",
        "pydantic>=2.0,<3.0",
        "pyyaml",
        "numpy",
        "scikit-learn",
        "pandas",
        "psutil",
    ],
    extras_require=EXTRAS,
    entry_points={
        "console_scripts": [
            "aiforge=aiforge_cli:main",
        ],
    },
    py_modules=["aiforge_cli"],
)
