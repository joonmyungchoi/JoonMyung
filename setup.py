import setuptools
from setuptools import find_packages
setuptools.setup(
    name="joonmyung",
    version="1.4.0",
    author="JoonMyung Choi",
    author_email="pizard@korea.ac.kr",
    description="JoonMyung's Library",
    url="https://github.com/pizard/JoonMyung.git",
    license="MIT",
    packages=find_packages(exclude=["playground",
                                    "playground.*",
                                    "99_backup",
                                    "99_backup.*",
                                    "*.egg-info"
                                    "*.egg-info.*"]),
    zip_safe=False,
    install_requires=[
    ]
)

# git add .
# git commit
# git push
# Remove *.egg-info
#
# python setup.py sdist; python -m twine upload dist/*
# ID:JoonmyungChoi