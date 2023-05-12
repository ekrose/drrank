import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='drrank',
    version='0.0.1',
    author='Evan K. Rose',
    author_email='ekrose@gmail.com',
    description='Implement the Empirical Bayes ranking scheme developed in Kline, Rose, and Walters (2023)',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/ekrose/drrank',
    install_requires=[
        'numpy',
        'pandas',
        'os',
        'gurobipy',
        'scipy'
      ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires='>=3.7',
)