import os
import setuptools


# Utility function to read a file at same level as this one.
def read(f_name):
    with open(os.path.join(os.path.dirname(__file__), f_name)) as f:
        return f.read()


def read_requirements():
    requirements = []
    for line in read('requirements.txt').split('\n'):
        if line and line[0] != '-' and line[0] != '#':
            requirements.append(line)
    return requirements


setuptools.setup(
    name='pandas-sigproc',
    version='1.0.1',
    description='Useful pandas extensions for detailed signal processing',
    long_description=read('README.md'),
    url='https://github.com/johns7591/pandas-sigproc',
    author='John Scanlon',
    packages=setuptools.find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=read_requirements()
)

