#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Python packages from requirements.txt
install_requirements() {
    echo "Installing Python libraries from requirements.txt..."
    python3.12 -m pip install -r requirements.txt
}

# Function to check and install specific Python packages
check_and_install_python_packages() {
    REQUIRED_LIBRARIES=(
        "logging"
        "yfinance"
        "numpy"
        "pandas"
        "os"
        "json"
        "datetime"
        "scipy"
        "matplotlib"
        "seaborn"
        "statsmodels"
        "pandas-datareader"
        "warnings"
    )

    echo "Checking Python libraries..."
    for lib in "${REQUIRED_LIBRARIES[@]}"; do
        echo "Checking $lib..."
        python3.12 -c "import ${lib}" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "$lib is not installed. Installing it..."
            python3.12 -m pip install $lib
            if [ $? -eq 0 ]; then
                echo "$lib installed successfully."
            else
                echo "Failed to install $lib. Please check your pip configuration."
                exit 1
            fi
        else
            echo "$lib is already installed."
        fi
    done
}

# Function to install Python 3.12
install_python3_12() {
    echo "Attempting to install Python 3.12..."
    OS=$(uname -s)
    case "$OS" in
        Linux*)
            echo "Installing Python 3.12 on Linux..."
            if command_exists apt; then
                sudo apt update && sudo apt install -y software-properties-common
                sudo add-apt-repository -y ppa:deadsnakes/ppa
                sudo apt update && sudo apt install -y python3.12 python3.12-venv python3.12-distutils
            elif command_exists yum; then
                sudo yum install -y gcc libffi-devel zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl-devel xz xz-devel libuuid-devel
                cd /usr/src
                sudo curl -O https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz
                sudo tar xzf Python-3.12.0.tgz
                cd Python-3.12.0
                sudo ./configure --enable-optimizations
                sudo make altinstall
            else
                echo "Unsupported Linux package manager. Please install Python 3.12 manually."
                exit 1
            fi
            ;;
        Darwin*)
            echo "Installing Python 3.12 on macOS..."
            if command_exists brew; then
                brew install python@3.12
            else
                echo "Homebrew is not installed. Please install Homebrew first (https://brew.sh) and try again."
                exit 1
            fi
            ;;
        *)
            echo "Unsupported operating system: $OS. Please install Python 3.12 manually."
            exit 1
            ;;
    esac

    # Verify if Python 3.12 was installed successfully
    if command_exists python3.12; then
        echo "Python 3.12 installed successfully."
    else
        echo "Failed to install Python 3.12. Please install it manually."
        exit 1
    fi
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

# Check if Python 3.12 is installed, if not, attempt to install it
if command_exists python3.12; then
    echo "Python 3.12 is installed."
else
    echo "Python 3.12 is not installed. Installing Python 3.12..."
    install_python3_12
fi

# Check if pip is installed
if command_exists pip3; then
    echo "pip is installed."
else
    echo "pip is not installed. Attempting to install pip..."
    python3.12 -m ensurepip --upgrade
fi

# Install required libraries from requirements.txt and check for additional imports
install_requirements
check_and_install_python_packages

# Run the Python script
echo "Running the Python script portfolio_optimization.py..."
python3.12 portfolio_optimization.py

echo "Execution completed."
