#!/usr/bin/env python

"""The setup script."""

import os

from setuptools import find_packages, setup

import nuxnet_package as module


def walker(base, *paths):
    file_list = set([])
    cur_dir = os.path.abspath(os.curdir)

    os.chdir(base)
    try:
        for path in paths:
            for dname, _dirs, files in os.walk(path):
                for filename in files:
                    file_list.add(os.path.join(dname, filename))
    finally:
        os.chdir(cur_dir)

    return list(file_list)


with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.rst', encoding='utf-8') as history_file:
    history = history_file.read()

with open('requirements.txt', encoding='utf-8') as req_file:
    requirements = req_file.read().splitlines()

setup(
    author='NuxNet contributors',
    author_email='noreply@example.org',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description='NuxNet 3D nuclei segmentation inference scaffold package.',
    entry_points={
        'console_scripts': [
            'nuxnet-pred=nuxnet_package.cli_pred:main',
        ],
    },
    install_requires=requirements,
    license='MIT',
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='nuclei segmentation inference 3d unet',
    name='nuxnet-inference',
    packages=find_packages(include=['nuxnet_package', 'nuxnet_package.*']),
    package_data={
        module.__name__: walker(
            os.path.dirname(module.__file__),
            'models',
            'data',
        ),
    },
    url='https://github.com/nuxnet/nuxnet-inference',
    version='2.0.0',
    zip_safe=False,
)
