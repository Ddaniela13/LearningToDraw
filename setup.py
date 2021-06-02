from setuptools import setup

setup(
    name='LearningToDraw',
    version='',
    packages=['model'],
    url='https://github.com/Ddaniela13/LearningToDraw',
    license='BSD 3-Clause',
    author='Daniela Mihai and Jonathon Hare',
    author_email='{adm1g15,jsh2}@soton.ac.uk',
    description='Sketch-based communication games between cooperative agents',
    entry_points={
        'console_scripts': ['commgame=commgame:main']
    }
)
