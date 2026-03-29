make

sizes=(16 81 169 256 400)

for var in "${sizes[@]}"
do
    sq=$(python3 -c "import math; print(int(math.sqrt($var)))")
    echo "================================================"
    echo "Testing with $var processes (${sq}x${sq} grid)"
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
