import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quickai",  # Replace with your own username
    version="1.3.7",
    author="geekjr",
    author_email="author@example.com",
    description="QuickAI is a Python library that makes it extremely easy to experiment with state-of-the-art "
                "Machine Learning models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geekjr/quickai",
    project_urls={
        "Bug Tracker": "https://github.com/geekjr/quickai/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["quickai/"],
    python_requires=">=3.6",
    install_requires=[
        'scikit-learn',
        'numpy',
        'matplotlib',
        'transformers'
    ]
)
