"""
Central dependency installation script for BDA Workshop notebooks.
This script provides a consistent way to install required packages across all notebooks.
"""

import subprocess
import sys
import os

def install_dependencies(requirements_file='requirements.txt', quiet=True):
    """
    Install dependencies from a requirements file.
    
    Args:
        requirements_file (str): Path to requirements.txt file
        quiet (bool): If True, suppress pip output
    
    Returns:
        bool: True if installation was successful, False otherwise
    """
    # Determine the absolute path to the requirements file
    # If we're in a notebook directory like minimized-enhanced/02-image, 
    # we need to go up to the root directory
    root_requirements = requirements_file
    if not os.path.isfile(requirements_file):
        # Try going up directories to find it
        current_dir = os.getcwd()
        workshop_root = current_dir
        
        # Check if we're in a module subdirectory like 02-image
        if os.path.basename(current_dir).startswith('0'):
            workshop_root = os.path.dirname(current_dir)
        
        # If we're in minimized-enhanced/02-image, go up again
        if os.path.basename(workshop_root) in ['minimized', 'minimized-enhanced', 'studio-workshop']:
            workshop_root = os.path.dirname(workshop_root)
        
        root_requirements = os.path.join(workshop_root, requirements_file)
    
    if not os.path.isfile(root_requirements):
        print(f"Requirements file not found at {root_requirements}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Searching in: {workshop_root}")
        return False
    
    print(f"Installing dependencies from {root_requirements}")
    
    # Build the pip install command
    pip_cmd = [sys.executable, '-m', 'pip', 'install', '-r', root_requirements]
    if quiet:
        pip_cmd.extend(['-q'])
    
    try:
        result = subprocess.run(pip_cmd, check=True)
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def install_specific_packages(packages, quiet=True):
    """
    Install specific packages.
    
    Args:
        packages (list): List of package specifiers (e.g., ["boto3>=1.37.4", "pillow"])
        quiet (bool): If True, suppress pip output
    
    Returns:
        bool: True if installation was successful, False otherwise
    """
    if not packages:
        print("No packages specified")
        return False
    
    # Build the pip install command
    pip_cmd = [sys.executable, '-m', 'pip', 'install'] + packages
    if quiet:
        pip_cmd.extend(['-q'])
    
    try:
        result = subprocess.run(pip_cmd, check=True)
        print(f"Packages installed successfully: {', '.join(packages)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

# Example usage for notebooks:
# from install_dependencies import install_dependencies
# install_dependencies()

if __name__ == "__main__":
    # If run directly, install dependencies from requirements.txt
    install_dependencies(quiet=False)
