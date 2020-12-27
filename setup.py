from setuptools import setup, find_packages

setup(
    name="facenet",
    version="0.0.1",
    description="Face recognition using TensorFlow",
    url="https://github.com/afeedh/facenet",
    packages=find_packages(),
    maintainer="Afeedh Shaji",
    maintainer_email="afeedhshaji98@gmail.com",
    include_package_data=True,
    license="MIT",
    install_requires=[
        "tensorflow==1.7",
        "scipy==1.1.0",
        "scikit-learn",
        "opencv-python==3.4.0.14",
        "h5py",
        "matplotlib",
        "Pillow",
        "requests",
        "psutil",
    ],
)