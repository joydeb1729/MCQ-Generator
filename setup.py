from setuptools import find_packages, setup

setup(
    name = 'MCQgenerator',
    version='0.1',
    author='Joydeb Gan Prokas',
    author_email='joydebganprokas@gmail.com',
    install_requires=['langchain','streamlit', 'python-dotenv', 'PyPDF2'],
    packages=find_packages()
)