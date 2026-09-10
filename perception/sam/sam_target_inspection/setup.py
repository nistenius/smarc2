import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sam_target_inspection'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ivan Stenius',
    maintainer_email='stenius@kth.se',
    description=('Adaptive close inspection: side-scan and forward-sonar target detection, '
                 'the candidate ledger, the inspection planner and the capture recorder'),
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sss_target_detector = sam_target_inspection.sss_target_detector:main',
            'fls_target_detector = sam_target_inspection.fls_target_detector:main',
            'inspection_planner = sam_target_inspection.inspection_planner_node:main',
            'inspection_recorder = sam_target_inspection.inspection_recorder:main',
        ],
    },
)
