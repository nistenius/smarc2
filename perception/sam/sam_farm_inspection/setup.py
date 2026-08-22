import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sam_farm_inspection'

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
    description='Algae-farm inspection: side-scan rope/buoy detection and farm localization',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sss_farm_detector = sam_farm_inspection.sss_farm_detector:main',
            'farm_localizer = sam_farm_inspection.farm_localizer_node:main',
            'farm_planner = sam_farm_inspection.farm_planner_node:main',
        ],
    },
)
