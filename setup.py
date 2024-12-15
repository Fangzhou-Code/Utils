from setuptools import setup, find_packages

setup(
    name="utils_rag",  # 替换为项目名称
    version="1.0.6",  # 项目版本
    description="Here is an enhanced version of the LocalEmbeddings class that provides similar functionality to the OpenAI embedding model and includes more features and detailed comments in English and Chinese",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="FangZhou",
    author_email="noah183225@gamil.com",
    url="https://github.com/Fangzhou-Code/Utils",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[],
)
