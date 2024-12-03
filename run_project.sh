#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Python packages
install_requirements() {
    echo "Installing Python libraries from requirements.txt..."
    python3.12 -m pip install -r requirements.txt
}

# Checking operating system
echo "Checking operating system..."
OS=$(uname -s)
case "$OS" in
    Linux*)
        echo "Operating System: Linux"
        ;;
    Darwin*)
        echo "Operating System: macOS"
        ;;
    *)
        echo "Unsupported operating system: $OS"
        exit 1
        ;;
esac

# Checking terminal specifications
echo "Checking terminal specifications..."
TERM=$(echo $TERM)
echo "Terminal type: $TERM"

# Check if Python 3.12 is installed
if command_exists python3.12; then
    echo "Python 3.12 is installed."
else
    echo "Python 3.12 is not installed. Please install Python 3.12 and try again."
    exit 1
fi

# Check if pip is installed
if command_exists pip3; then
    echo "pip is installed."
else
    echo "pip is not installed. Attempting to install pip..."
    python3.12 -m ensurepip --upgrade
fi

# Install required libraries
install_requirements

# Run the Python script
echo "Running the Python script main_V3.py..."
python3.12 main_V3.py

echo "Execution completed."