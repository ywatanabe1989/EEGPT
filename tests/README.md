# Working with pytest

This guide provides an overview of how to configure and use `pytest` for testing your Python code. Below are the key files and directories involved in setting up `pytest` for your project.

## pytest.ini
This is the configuration file for `pytest`. It allows you to define default command-line options, custom markers, test paths, and other settings that `pytest` will use when running your tests.

Example `pytest.ini`:
```ini
[pytest]
python_paths = ./
testspaths = ./tests
```

## tests/pytestconf.py
This file is used to configure `pytest` fixtures, hooks, and plugins. You can define setup and teardown logic for your tests, customize test reports, and integrate with other systems.

Example `tests/conftest.py`:
```python
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
```

## Mirror scripts under tests dir
Your test files should mirror the structure of your project's scripts. For each script in your project, there should be a corresponding test file in the `tests` directory.

Project structure:
```
project/
    scripts/
        example.py
    tests/
        test_example.py
```

mirror_sciripts.sh

``` bash
#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="./scripts/"
DEST_DIR="./tests/scripts/"

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Create the destination directory if it does not exist
if [ ! -d "$DEST_DIR" ]; then
    mkdir -p "$DEST_DIR"
fi

# Use find to iterate over all files in the source directory
find "$SOURCE_DIR" -type f -name "*.py" -exec bash -c ' \
    SOURCE_DIR="$0"; \
    DEST_DIR="$1"; \
    FILE_PATH="$2"; \
    FILE_NAME="${FILE_PATH##*/}"; \
    # Check if the file already has an underscore prefix
    if [[ "$FILE_NAME" == _* ]]; then
        DEST_FILE_NAME="$FILE_NAME"
    else
        DEST_FILE_NAME="_$FILE_NAME"
    fi
    DEST_SUBDIR="${FILE_PATH#$SOURCE_DIR}"; \
    DEST_SUBDIR="${DEST_SUBDIR%$FILE_NAME}"; \
    DEST_PATH="$DEST_DIR$DEST_SUBDIR"; \
    mkdir -p -- "$DEST_PATH"; \
    # Copy the file, using the determined destination file name
    cp -- "$FILE_PATH" "$DEST_PATH/$DEST_FILE_NAME"' "$SOURCE_DIR" "$DEST_DIR" {} \;

# Inform the user that the process is complete
echo "Mirroring complete with underscore prefix for files."
```

## tests/test_example.py
This is an example of a test file. It should contain your test cases and use assertions to verify the behavior of the code you are testing.

Example `tests/test_example.py`:
```python
import pytest
from scripts.example import add

def test_add():
    assert add(2, 3) == 5
```

By following this structure and these examples, you can set up `pytest` to run automated tests for your project. Make sure to install `pytest` using `pip` if you haven't already:

```bash
pip install pytest pytest-xdist
```

To run your tests, simply execute the `pytest` command in your terminal:

```bash
pytest -n 10
```

For more advanced usage, refer to the [pytest documentation](https://docs.pytest.org/en/stable/).
