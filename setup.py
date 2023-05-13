import site
import sys
from setuptools import find_packages, setup

site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setup(
    name="mutinfo",
    version="0.0.1",
    python_requires=">=3.6.9",
    install_requires=[
        "numpy >= 1.1.0",
        "scipy >= 1.5.4",
        "scikit-learn >= 0.24.2"
    ],
    dependency_links = [
    ],
    packages=find_packages(where="source/python"),
    package_dir={"": "source/python"},
    author="Ivan Butakov",
    author_email="vanessbut@yandex.ru",
    description="Mutual information toolbox",
)
