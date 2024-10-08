from setuptools import find_packages, setup

setup(
    name="irrevolutions",
    version="2024.0.1",
    package_dir={"": "src"},  # The root package is under the 'src' directory
    packages=find_packages("src"),  # Find packages under the 'src' directory
    include_package_data=True,
    description="A Python package for solving nonlinear and nonconvex evolutionary problems using a general energetic notion of stability and dolfinx.",
    author="Andrés A León Baldelli",
    author_email="leon.baldelli@cnrs.fr",
    url="https://github.com/kumiori3/irrevolutions",  # Replace with your repo

    package_data={
        "irrevolutions.models": ["default_parameters.yml"],
    },
    python_requires=">=3.9",

)
