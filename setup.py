import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NonlinearML",
    version="0.0.1",
    author="Siinn Che",
    author_email="siinn.che@alliancebernstein.com",
    description="AB Nonlinear factors ML project",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=['']
)

