 rm -rf ./*.egg-info ./dist/*
 python3 setup.py sdist; python -m twine upload dist/*
