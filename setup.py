from setuptools import setup, find_packages

setup(
    name='train_model_image_package',
    version='0.1',
    packages=find_packages(exclude=['data', 'myenv', 'venv']),
    description='No description',
    author='Duy',
    author_email='hoangtanduynx@gmail.com',
    url='https://github.com/HoangDuy09042001/TrainModelPackage.git',
)