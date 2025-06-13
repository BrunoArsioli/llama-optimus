from setuptools import setup, find_packages

setup(
    name="llama-optimus",
    version="0.1.0",
    description="Bayesian optimization for llama.cpp performance flags.",
    author="Bruno Arsioli",
    packages=find_packages(),
    install_requires=[          # can read from requirements.txt
        "optuna>=3.0",
        "pandas",
    ], 
    entry_points={
        'console_scripts': [
            'llama-optimus = llama_optimus.cli:main',
        ],
    },
    python_requires='>=3.8',
)
