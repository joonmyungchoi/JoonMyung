import setuptools

setuptools.setup(
    name="joonmyung",
    version="0.1.0",
    author="JoonMyung Choi",
    author_email="pizard@korea.ac.kr",
    description="JoonMyung's Library",
    url="https://github.com/pizard/JoonMyung.git",
    license="MIT",
    py_modules=['utils'],
    packages=setuptools.find_packages(),
    zip_safe=False,
    install_requires=[

    ]
)
# python setup.py sdist
# python -m twine upload dist/*
# ID:JoonmyungChoi