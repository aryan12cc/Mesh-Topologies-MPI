#include <mpi.h>
#include <iostream>
#include <cmath>

#define N (1 << 24)  

int get_rank_3d(MPI_Comm comm, int x, int y, int z) {
    int coords[3] = {x, y, z};
    int rank;
    MPI_Cart_rank(comm, coords, &rank);
    return rank;
}

void bcast_xyz_naive(int* data, int count, MPI_Datatype datatype, MPI_Comm comm, int dims[3]) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int coords[3];
    MPI_Cart_coords(comm, rank, 3, coords);
    int my_x = coords[0];
    int my_y = coords[1];
    int my_z = coords[2];

    int dim_x = dims[0];
    int dim_y = dims[1];
    int dim_z = dims[2];

    MPI_Status status;

    if (my_y == 0 && my_z == 0) {
        if (my_x > 0) {
            int source = get_rank_3d(comm, my_x - 1, my_y, my_z);
            MPI_Recv(data, count, datatype, source, 1, comm, &status);
        }
        
        if (my_x < dim_x - 1) {
            int dest = get_rank_3d(comm, my_x + 1, my_y, my_z);
            MPI_Send(data, count, datatype, dest, 1, comm);
        }
    }

    if (my_z == 0) {
        if (my_y > 0) {
            int source = get_rank_3d(comm, my_x, my_y - 1, my_z);
            MPI_Recv(data, count, datatype, source, 2, comm, &status);
        }
        
        if (my_y < dim_y - 1) {
            int dest = get_rank_3d(comm, my_x, my_y + 1, my_z);
            MPI_Send(data, count, datatype, dest, 2, comm);
        }
    }

    if (my_z > 0) {
        int source = get_rank_3d(comm, my_x, my_y, my_z - 1);
        MPI_Recv(data, count, datatype, source, 3, comm, &status);
    }

    if (my_z < dim_z - 1) {
        int dest = get_rank_3d(comm, my_x, my_y, my_z + 1);
        MPI_Send(data, count, datatype, dest, 3, comm);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int dim_len = round(cbrt(world_size));
    if (dim_len * dim_len * dim_len != world_size) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) std::cerr << "Error: Need cubic number of procs." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int dims[3] = {dim_len, dim_len, dim_len};
    int periods[3] = {0, 0, 0};
    int reorder = 1;
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cart_comm);

    int my_rank;
    int my_coords[3];
    MPI_Comm_rank(cart_comm, &my_rank);
    MPI_Cart_coords(cart_comm, my_rank, 3, my_coords);

    int* data_array = new int[N];
    for (int i = 0; i < N; i++) {
        data_array[i] = 0;
    }
    
    if (my_coords[0] == 0 && my_coords[1] == 0 && my_coords[2] == 0) {
        for (int i = 0; i < N; i++) {
            data_array[i] = i;
        }
    }

    MPI_Barrier(cart_comm);
    
    double start = MPI_Wtime();
    
    bcast_xyz_naive(data_array, N, MPI_INT, cart_comm, dims);
    
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
        printf("ERROR: Rank %d (Coords %d,%d,%d) failed to receive correct data.\n", 
               my_rank, my_coords[0], my_coords[1], my_coords[2]);
    }
    
    if (my_rank == 0) {
        printf("Naive 3D Broadcast (N=%d) Finished in %f seconds.\n", N, max_end - start);
    }

    delete[] data_array;
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
