from glob import glob
import os

from setuptools import find_packages, setup

package_name = "roundabout_pkg"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="MAE148 Team 12",
    maintainer_email="team12@ucsd.edu",
    description="Roundabout right-of-way negotiation for UCSD Robocar (ROS2)",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "lane_follower_node = roundabout_pkg.nodes.lane_follower_node:main",
            "yield_detector_node = roundabout_pkg.nodes.yield_detector_node:main",
            "car_tracker_node = roundabout_pkg.nodes.car_tracker_node:main",
            "right_of_way_node = roundabout_pkg.nodes.right_of_way_node:main",
            "cmd_vel_mux_node = roundabout_pkg.nodes.cmd_vel_mux_node:main",
            "scenario_sim = roundabout_pkg.sim.scenario_sim:main",
        ],
    },
)
