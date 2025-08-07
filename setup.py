from setuptools import setup

setup(
    name="vickrey",
    version="0",
    packages=["vickrey"],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "jax",
        "geopandas",
        "quadax",
        "jaxopt"
    ]
)
