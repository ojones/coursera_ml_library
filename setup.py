try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Python library for machine learning',
    'author': 'Oswald Jones',
    'url': 'TODO.',
    'download_url': 'TODO.',
    'author_email': 'wakeupoj@gmail.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['ml_libraray'],
    'scripts': [],
    'name': 'ml_libraray'
}

setup(**config)