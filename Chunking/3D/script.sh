#!/bin/bash

echo "Compiling 3D broadcast implementations..."
mpic++ naive_unserialized.cpp -o naive_unserialized
mpic++ chunked_serialized.cpp -o chunked_serialized
echo "Done!"
echo ""

var=64
echo "=========================================="
echo "Testing with $var processes (4x4x4 grid)"
echo "=========================================="
echo ""

echo "Naive 3D Broadcast (sends all data at once, no serialization):"
mpirun --oversubscribe -np $var ./naive_unserialized
echo ""

echo "Chunked 3D Serialized Broadcast (sends data in chunks with pipelining and serialization):"
mpirun --oversubscribe -np $var ./chunked_serialized
echo ""
