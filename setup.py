from setuptools import setup, find_packages

setup(
    name='irrevolutions',
    version='2024.0.1',
    package_dir={'': 'src'},  # The root package is under the 'src' directory
    packages=find_packages('src'),  # Find packages under the 'src' directory
)
