from setuptools import setup, find_packages

setup(
    name='irrevolution',
    version='2024.0.1',
    packages=find_packages('src'),  # Find packages under the 'src' directory
    package_dir={'': 'src'},  # Specify that the root package is under the 'src' directory
)
