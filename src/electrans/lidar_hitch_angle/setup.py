from setuptools import find_packages, setup

package_name = 'lidar_hitch_angle'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            [
                'config/lidar_hitch_angle.param.yaml',
                'config/trailer_self_filter.param.yaml',
            ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hlc',
    maintainer_email='mtuer@uwaterloo.ca',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'hitchangle=lidar_hitch_angle.lidar_hitch_angle:main',
        'hitchangle_filtered=lidar_hitch_angle.hitch_filter_vis:main',
        'trailer_self_filter=lidar_hitch_angle.trailer_self_filter:main',
        ],
    },
)
