from setuptools import find_packages, setup

with open("requirements.txt") as file:
    requirements = file.read().splitlines()

setup(
    name="vqvae",
    author="Ryuu You",
    description="Deep learning fast prototyping template for VQ-VAE and related models.",
    url="https://github.com/yourname/vqvae",
    install_requires=requirements,
    packages=find_packages(where="lib"),
    package_dir={"": "lib"},
    python_requires=">=3.8",
)