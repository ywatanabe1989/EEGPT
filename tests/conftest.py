#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-05 22:30:41 (ywatanabe)"


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

    yield  # Run the tests

    # After tests, reset the IS_DEBUG.yaml to its original state
    with open(config_path, "w") as file:
        yaml.dump(original_config, file, default_flow_style=False)
