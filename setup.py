from setuptools import setup

setup(
    name="vickrey",
    version="0",
    packages=["vickrey"],
    python_requires=">=3.7",
    package_data={"vickrey": ["data/*"]},
    install_requires=[
        "numpy",
        "pandas",
        "tables",
        "scipy",
        "matplotlib",
        "jax",
        "geopandas",
        "quadax",
        "jaxopt"
    ]
)
