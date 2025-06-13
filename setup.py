from setuptools import setup, find_packages

setup(
    name='face-mask-detection',
    version='1.0',
    description='Real-time face mask detection using deep learning.',
    author='Komali Koppaka',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.0.0',
        'opencv-python',
        'imutils',
        'matplotlib',
        'numpy',
    ],
    python_requires='>=3.6',
)
