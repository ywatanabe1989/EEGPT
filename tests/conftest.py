#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-08 19:42:52 (ywatanabe)"

import logging

import pytest
import yaml


@pytest.fixture(scope="session", autouse=True)
def set_debug_mode():
    """Fixture to manage debug mode during tests."""
    config_path = "./config/IS_DEBUG.yaml"

    # Read the original state from IS_DEBUG.yaml
    with open(config_path, "r") as file:
        original_config = yaml.safe_load(file)

    # Set IS_DEBUG to True for testing
    debug_config = {"IS_DEBUG": True}
    with open(config_path, "w") as file:
        yaml.dump(debug_config, file, default_flow_style=False)
    logging.warning(
        f"\nCONFIG is set to {debug_config['IS_DEBUG']} for pytest"
    )
    # print(f"\nCONFIG is set to {debug_config['IS_DEBUG']} for pytest")

    yield  # Run the tests

    # After tests, reset IS_DEBUG.yaml to its original state
    with open(config_path, "w") as file:
        yaml.dump(original_config, file, default_flow_style=False)
    # print(
    #     f"\nCONFIG has been reverted to {original_config['IS_DEBUG']} after pytest"
    # )
    logging.warning(
        f"\nCONFIG has been reverted to {original_config['IS_DEBUG']} after pytest"
    )


# import pytest
# import yaml


# @pytest.fixture(scope="session", autouse=True)
# def set_debug_mode():
#     """Fixture to manage debug mode during tests."""
#     config_path = "./config/IS_DEBUG.yaml"

#     # Read the original state from IS_DEBUG.yaml
#     with open(config_path, "r") as file:
#         original_config = yaml.safe_load(file)

#     # Set IS_DEBUG to True for testing
#     debug_config = {"IS_DEBUG": True}
#     with open(config_path, "w") as file:
#         yaml.dump(debug_config, file, default_flow_style=False)
#     print(f"\nCONFIG is set as {debug_config['IS_DEBUG']} by pytest")

#     yield  # Run the tests

#     # After tests, reset the IS_DEBUG.yaml to its original state
#     with open(config_path, "w") as file:
#         yaml.dump(original_config, file, default_flow_style=False)
#     print(f"\nCONFIG is reverted as {original_config['IS_DEBUG']} by pytest")
