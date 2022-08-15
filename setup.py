import os
import re
from setuptools import setup


def find_packages(package, basepath):
    packages = [package]
    for name in os.listdir(basepath):
        path = os.path.join(basepath, name)
        if not os.path.isdir(path):
            continue
        packages.extend(find_packages('%s.%s'%(package, name), path))
    return packages


here = os.path.abspath(os.path.dirname(__file__))
desc = 'Multivariate function minimizer based on the tensor train approach.'
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    desc_long = f.read()


with open(os.path.join(here, 'ttopt/__init__.py'), encoding='utf-8') as f:
    text = f.read()
    version = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", text, re.M)
    version = version.group(1)


setup_args = dict(
    name='ttopt',
    version=version,
    description=desc,
    long_description=desc_long,
    long_description_content_type='text/markdown',
    author='Andrei Chertkov',
    author_email='a.chertkov@skoltech.ru',
    url='https://github.com/SkoltechAI/ttopt',
    classifiers=[
        'Development Status :: 4 - Beta', # 3 - Alpha, 5 - Production/Stable
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='function optimization function minimization low-rank representation tensor train format TT-decomposition cross approximation',
    packages=find_packages('ttopt', './ttopt/'),
    python_requires='>=3.6',
    project_urls={
        'Source': 'https://github.com/SkoltechAI/ttopt',
    },
)


if __name__ == '__main__':
    setup(
        **setup_args,
        # Please, install it manually:
        # install_requires=['numpy', 'scipy', 'maxvolpy'],
        include_package_data=True)
