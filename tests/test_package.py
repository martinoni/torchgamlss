import importlib.metadata

import torchgamlss


def test_runtime_version_matches_installed_package_metadata():
    assert torchgamlss.__version__ == importlib.metadata.version("torchgamlss")


def test_alpha_version_is_not_the_package_skeleton_placeholder():
    assert torchgamlss.__version__ != "0.0.0"
