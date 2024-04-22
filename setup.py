import os

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)


def _load_requirements(path_dir: str, file_name: str = "requirements.txt") -> list:
    reqs = parse_requirements(open(os.path.join(path_dir, file_name)).readlines())
    return list(map(str, reqs))


setup(
    name="sd_inference",
    version="0.0.2",
    description="",
    packages=find_packages(
        where="src",  # '.' by default
        include=["stable_diffusion_inference", "ldm*"],
    ),
    package_dir={"": "src"},
    install_requires=_load_requirements(_PATH_ROOT),
    include_package_data=True,
)
