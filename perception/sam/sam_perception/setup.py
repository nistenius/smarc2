import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sam_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ivan Stenius',
    maintainer_email='stenius@kth.se',
    description='SAM 2.2 perception sensors: sim monitor / hardware drivers (Sonar 3D-15, RealSense D435i)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_monitor = sam_perception.perception_monitor:main',
        ],
    },
)
