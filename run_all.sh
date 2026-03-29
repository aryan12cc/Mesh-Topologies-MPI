#!/bin/bash
set -e

# Build all C++ MPI programs
echo "Building 2D Bcast..."
cd 2D/Broadcast && make && cd ../..

echo "Building 2D Reduce..."
cd 2D/Reduce && make && cd ../..

echo "Building 3D Bcast..."
cd 3D/Broadcast && make && cd ../..

echo "Building 3D Reduce..."
cd 3D/Reduce && make && cd ../..

echo "Building Chunking 2D..."
cd Chunking/2D && make && cd ../..

echo "Building Chunking 3D..."
cd Chunking/3D && make && cd ../..

echo "Build complete."

# Run all MPI experiments
echo "Running 2D Broadcast experiments..."
cd 2D/Broadcast && bash script.sh && cd ../..

echo "Running 2D Reduce experiments..."
cd 2D/Reduce && bash script.sh && cd ../..

echo "Running 3D Broadcast experiments..."
cd 3D/Broadcast && bash script.sh && cd ../..

echo "Running 3D Reduce experiments..."
cd 3D/Reduce && bash script.sh && cd ../..

echo "Running Chunking 2D experiments..."
cd Chunking/2D && bash script.sh && cd ../..

echo "Running Chunking 3D experiments..."
cd Chunking/3D && bash script.sh && cd ../..

echo "All experiment scripts completed."

# Generate Plots
echo "Generating plots..."
if [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python3"
fi

$PYTHON_CMD visualize_performance.py

echo "Generating chunking plots..."
cd Chunking
if [ -d "../.venv" ]; then
    PYTHON_CMD="../.venv/bin/python"
    # Or just use the same python logic but account for path
else
    PYTHON_CMD="python3"
fi
$PYTHON_CMD visualize_chunk_variation.py
cd ..

echo "Done. Check the 'plots' and 'Chunking/chunking_plots' directories."
