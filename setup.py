#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Deep Multimodal Learning for Winding Fault Diagnosis",
    author="Vishnudev Krishnadas",
    author_email="",
    url="https://github.com/andi677/AI-FAPS_Vishnudev_Krishnadas",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
