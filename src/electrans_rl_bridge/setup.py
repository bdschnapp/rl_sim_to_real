from setuptools import find_packages, setup

package_name = "electrans_rl_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/electrans_rl_bridge.launch.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ben Schnapp",
    maintainer_email="ben.schnapp@cplaneai.com",
    description="Lane reference + e2e_rl TD3 inference bridge.",
    license="Apache License 2.0",
    entry_points={
        "console_scripts": [
            "lane_reference_node = electrans_rl_bridge.lane_reference_node:main",
            "rl_bridge_node = electrans_rl_bridge.rl_bridge_node:main",
        ],
    },
)
