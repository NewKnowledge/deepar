from setuptools import setup, find_packages

setup(
    name="deepar",
    version="0.0.3",
    description="DeepAR tensorflow 2 implementation",
    author="Jeffrey Gleason",
    author_email="jeffrey.gleason@yonder.co",
    url="https://github.com/NewKnowledge/deepar/tree/master",
    install_requires=[
        #"tensorflow-gpu==2.0.0", 
        "tensorflow",
        "numpy>=1.15.4",
        "pandas>=0.23.4",
        "scikit-learn[alldeps]>=0.20.3",
    ],
    packages=find_packages(),
    extras_require={
        "dev": ["pytest", "jupyter", "fire"],
    },
)

