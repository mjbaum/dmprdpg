from setuptools import setup

setup(
	name="dmprdpg",
	version="0.1",
	packages=[
		"dmprdpg",
	],
	install_requires=[
		"numpy",
		"scipy",
        "networkx",
        "statsmodels",
        "matplotlib"
        "scikit-learn"
	],
)
