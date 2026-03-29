#include <mpi.h>
#include <iostream>
#include <cmath>

#define N (1 << 24)

int get_rank_from_coords(MPI_Comm comm, int row, int col) {
    int coords[2] = {row, col};
    int rank;
    MPI_Cart_rank(comm, coords, &rank);
    return rank;
}

void bcast_row_col(int* data, int count, MPI_Datatype datatype,
                   MPI_Comm cart_comm, int my_row, int my_col, int dims[2]) {
    int rows = dims[0];
    int cols = dims[1];
    MPI_Status status;

    if (my_row == 0) {
        if (my_col == 0) {
            if (cols > 1) {
                int east_neighbor = get_rank_from_coords(cart_comm, 0, my_col + 1);
                MPI_Send(data, count, datatype, east_neighbor, 0, cart_comm);
            }
        } else {
            int west_neighbor = get_rank_from_coords(cart_comm, 0, my_col - 1);
            MPI_Recv(data, count, datatype, west_neighbor, 0, cart_comm, &status);

            if (my_col < cols - 1) {
                int east_neighbor = get_rank_from_coords(cart_comm, 0, my_col + 1);
                MPI_Send(data, count, datatype, east_neighbor, 0, cart_comm);
            }
        }
    }

    if (my_row == 0) {
        if (rows > 1) {
            int south_neighbor = get_rank_from_coords(cart_comm, my_row + 1, my_col);
            MPI_Send(data, count, datatype, south_neighbor, 1, cart_comm);
        }
    } else {
        int north_neighbor = get_rank_from_coords(cart_comm, my_row - 1, my_col);
        MPI_Recv(data, count, datatype, north_neighbor, 1, cart_comm, &status);

        if (my_row < rows - 1) {
            int south_neighbor = get_rank_from_coords(cart_comm, my_row + 1, my_col);
            MPI_Send(data, count, datatype, south_neighbor, 1, cart_comm);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int dim_len = sqrt(world_size);
    if (dim_len * dim_len != world_size) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) std::cerr << "Error: Need square number of procs." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int dims[2] = {dim_len, dim_len};
    int periods[2] = {0, 0};
    int reorder = 1;
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    int my_rank;
    int my_coords[2];
    MPI_Comm_rank(cart_comm, &my_rank);
    MPI_Cart_coords(cart_comm, my_rank, 2, my_coords);

    int* data_array = new int[N];
    for (int i = 0; i < N; i++) {
        data_array[i] = 0;
    }

    if (my_coords[0] == 0 && my_coords[1] == 0) {
        for (int i = 0; i < N; i++) {
            data_array[i] = i;
        }
    }

    MPI_Barrier(cart_comm);

    double start = MPI_Wtime();

    bcast_row_col(data_array, N, MPI_INT, cart_comm, my_coords[0], my_coords[1], dims);

    double end = MPI_Wtime();

    double max_end;
    MPI_Reduce(&end, &max_end, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    bool error = false;
    for (int i = 0; i < N; i++) {
        if (data_array[i] != i) {
            error = true;
            break;
        }
    }

    if (error) {
        printf("ERROR: Rank %d (Coords %d,%d) failed to receive correct data.\n", my_rank, my_coords[0], my_coords[1]);
    }

    if (my_rank == 0) {
        printf("Naive Row-Column Broadcast (N=%d) Finished in %f seconds.\n", N, max_end - start);
    }

    delete[] data_array;
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
