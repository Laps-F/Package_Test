from setuptools import setup


import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")



setup(
    name='lib',
    version=get_version("lib/__init__.py"),
    description='Just a test of package creation',
    url='https://github.com/Laps-F/Package_Test',
    author='Mauro Santos',
    author_email='maurolaps@gmail.com',
    packages=['lib'],
    install_requires=[]
)