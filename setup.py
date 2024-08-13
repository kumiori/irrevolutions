from setuptools import find_packages, setup

setup(
    name="irrevolutions",
    version="2024.0.1",
    package_dir={"": "src"},  # The root package is under the 'src' directory
    packages=find_packages("src"),  # Find packages under the 'src' directory
    include_package_data=True,
    package_data={
        "irrevolutions.models": ["default_parameters.yml"],
    },
)
