from setuptools import setup
import os
from glob import glob

package_name = 'riptide_controllers2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # (os.path.join('share', package_name, 'cfg'), glob('cfg/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mitch',
    maintainer_email='sayremitch@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller = riptide_controllers2.controller_working:main',
            'thruster_solver = riptide_controllers2.thruster_solver:main',
        ],
    },
    py_modules=[
        'riptide_controllers2.Controllers'
    ]
)
