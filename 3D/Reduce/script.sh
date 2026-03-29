make

sizes=(27 64 216 343 729)

for var in "${sizes[@]}"
do
    cb=$(python3 -c "import math; print(int(round(math.pow($var, 1.0/3.0))))")
    echo "================================================"
    echo "Testing with $var processes (${cb}x${cb}x${cb} grid)"
    echo "================================================"

    echo "brute"
    mpirun --oversubscribe -np $var brute

    echo ""
    echo "bfs"
    mpirun --oversubscribe -np $var bfs

    echo ""
    echo "recursive"
    mpirun --oversubscribe -np $var recursive

    echo ""
    echo "row_pipeline"
    mpirun --oversubscribe -np $var row_pipeline

    echo ""
    echo "sqrt"
    mpirun --oversubscribe -np $var sqrt

    echo ""
done
