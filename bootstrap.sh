#!/bin/bash

# Define the name of the virtual environment folder
VENV_NAME="cs7180"

# Update pip to the latest version
python3 -m pip install --upgrade pip

# Create the virtual environment
python3 -m venv $VENV_NAME

# Activate the environment
source $VENV_NAME/bin/activate

# Install packages using the requirements file
pip install -r requirements.txt

echo "Environment setup is complete."
