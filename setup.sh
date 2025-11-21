#!/bin/bash

echo "Setting up VCU-Bridge environment..."

# Create conda environment
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate conda environment
echo "Activating environment..."
conda activate vcu-bridge

# Install the project in editable mode
echo "Installing VCU-Bridge package..."
pip install -e .

echo "Setup complete! Activate the environment with: conda activate vcu-bridge"
