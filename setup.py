from setuptools import find_packages, setup



setup(
    name="fighingcv",
    version="1.0.0",
    author="xmu-xiaoma666",
    author_email="julien@huggingface.co",
    description=(
        "FightingCV Codebase For Attention,Backbone, MLP, Re-parameter, Convolution"
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords=(
        "Attention"
        "Backbone"
    ),
    license="Apache",
    url="https://github.com/xmu-xiaoma666/External-Attention-pytorch",
    package_dir={"": "."},
    packages=find_packages("."),
    # entry_points={
    #     "console_scripts": [
    #         "huggingface-cli=huggingface_hub.commands.huggingface_cli:main"
    #     ]
    # },
    python_requires=">=3.7.0",
    # install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)