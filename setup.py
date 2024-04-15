from setuptools import setup, find_packages

setup(
    name='Multimodel video classification',
    version='0.1.0',
    author='Shamim Ibne Shahid',
    author_email='titu2297@yahoo.com',
    packages=find_packages(),
    description= 'Multimodel video classification',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'torch',
        'openai-whisper',  # This is typically how Whisper might be added, check PyPI for the exact package name
        'scikit-learn',    # 'sklearn' is part of scikit-learn
        'pickle-mixin',    # 'pickle' is part of the standard library but may need mixins
        # 'dataclasses',   # Uncomment this line if your Python version is < 3.7
    ],
    python_requires='>=3.7',  # Assuming you're using features from Python 3.7+
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
     
)
