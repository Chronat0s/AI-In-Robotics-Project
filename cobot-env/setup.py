from setuptools import setup

setup(
    name="cobot_ai4robotics",
    version='0.1',
    install_requires=['gym',
                      'pybullet',
                      'numpy',
                      'matplotlib',
                      'torch',
                      'pandas',
                      'ultralytics'],
    package_data={'cobot_ai4robotics': ['resources/*.urdf', 
                                        'resources/projectiles/ycb_objects/*/*',
                                        'resources/cobot/*'
                                        ]}
)
