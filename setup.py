import setuptools

requirements = []
with open("requirements.txt", "r") as fh:
    for line in fh:
        requirements.append(line.strip())

setuptools.setup(install_requires=requirements)
