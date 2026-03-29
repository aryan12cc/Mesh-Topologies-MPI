#!/bin/bash

echo "Compiling broadcast implementations..."
mpic++ naive_unserialized.cpp -o naive_unserialized
mpic++ chunked_serialized.cpp -o chunked_serialized
echo "Done!"
echo ""

var=64
echo "=========================================="
echo "Testing with $var processes (8x8 grid)"
echo "=========================================="
echo ""

echo "Naive Broadcast (sends all data at once, no serialization):"
mpirun --oversubscribe -np $var ./naive_unserialized
echo ""

echo "Chunked Serialized Broadcast (sends data in chunks with pipelining and serialization):"
mpirun --oversubscribe -np $var ./chunked_serialized
echo ""
