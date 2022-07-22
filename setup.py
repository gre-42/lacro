#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The setup script."""


from setuptools import find_packages, setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

install_requires = [
    'simpleeval'
]

extras_require = {
    'gui': ['ncurses']
}

setup_requires = [
]

tests_require = [
]


def files(dir):
    import glob
    import os
    return [f.replace('/', os.sep) for f in glob.glob(os.path.join(dir, '*'))
            if os.path.isfile(f) and os.path.basename(f) != '__init__.py']


def scripts(dir):
    res = [f for f in files(dir) if not f.endswith('.py')]
    if len(res) == 0:
        raise ValueError(f'Could not find a single script in {dir}')
    return res


def entry_point(filename):
    import os.path
    import re
    if not os.path.exists(filename):
        raise ValueError(f'File {filename} does not exist')
    f = re.match(r'^(.*)\.py$', filename).group(1)
    return f'{os.path.basename(f)}={f.replace("/", ".")}:main'


def entry_points(dir):
    res = [entry_point(f) for f in files(dir) if f.endswith('.py')]
    if len(res) == 0:
        raise ValueError(f'Could not find a single entrypoint in {dir}')
    return res


setup(
    name='lacro',
    version='0.1.0',
    description="Parser for a LaTeX-like syntax",
    long_description=readme + '\n\n' + history,
    author="Marc Kramer",
    url='https://github.com/gre-42/lacro',
    packages=find_packages(),
    data_files=[
    ],
    scripts=[
    ],
    entry_points={
        'console_scripts': (
            entry_points('lacro/app'),
            ),
    },
    test_suite='tests',
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    setup_requires=setup_requires,
    tests_require=tests_require,
    license="MIT license",
    zip_safe=False,
    keywords='lacro',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
