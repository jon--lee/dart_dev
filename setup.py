from setuptools import setup

setup(name='dart',
      version='0.1.0',
      install_requires=["gym==0.8.0",
                        "numpy",
                        "scipy",
                        "matplotlib",
                        "pandas",
                        "sklearn",
                        "keras==2.0.4", 
                        "tensorflow==1.1.0",
                        "mujoco-py==0.5.7"]
)
