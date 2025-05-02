import setuptools
from setuptools import find_packages

def fetch_requirements(filename):
    with open(filename) as f:
        return [ln.strip() for ln in f.read().split("\n")]

setuptools.setup(
    name="joonmyung",
    version="1.6.1",
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
        # fetch_requirements("requirements.txt"),
    ]
)

# git add .
# git commit
# git push

# rm -rf ./*.egg-info ./dist/*; python3 setup.py sdist; python -m twine upload dist/*

# ID:JoonmyungChoi