#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

int message_count = 0;  

int get_rank_3d(MPI_Comm comm, int x, int y, int z) {
    int coords[3] = {x, y, z};
    int rank;
    MPI_Cart_rank(comm, coords, &rank);
    return rank;
}

void reduce_xyz_pipeline(int* local_val, int* global_result, int root_rank, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int dims[3], periods[3], coords[3];
    MPI_Cart_get(comm, 3, dims, periods, coords);
    int my_x = coords[0];
    int my_y = coords[1];
    int my_z = coords[2];
    int dim_x = dims[0];
    int dim_y = dims[1];
    int dim_z = dims[2];

    int current_sum = *local_val;
    MPI_Status status;

    if (my_z < dim_z - 1) {
        int back_neighbor = get_rank_3d(comm, my_x, my_y, my_z + 1);
        int received_val;
        MPI_Recv(&received_val, 1, MPI_INT, back_neighbor, 30, comm, &status);
        current_sum += received_val;
    }

    if (my_z > 0) {
        int front_neighbor = get_rank_3d(comm, my_x, my_y, my_z - 1);
        MPI_Send(&current_sum, 1, MPI_INT, front_neighbor, 30, comm);
        message_count++;  
    }

    if (my_z == 0) {
        
        if (my_y < dim_y - 1) {
            int south_neighbor = get_rank_3d(comm, my_x, my_y + 1, my_z);
            int received_val;
            MPI_Recv(&received_val, 1, MPI_INT, south_neighbor, 20, comm, &status);
            current_sum += received_val;
        }

        if (my_y > 0) {
            int north_neighbor = get_rank_3d(comm, my_x, my_y - 1, my_z);
            MPI_Send(&current_sum, 1, MPI_INT, north_neighbor, 20, comm);
            message_count++;  
        }
    }

    if (my_z == 0 && my_y == 0) {
        
        if (my_x < dim_x - 1) {
            int east_neighbor = get_rank_3d(comm, my_x + 1, my_y, my_z);
            int received_val;
            MPI_Recv(&received_val, 1, MPI_INT, east_neighbor, 10, comm, &status);
            current_sum += received_val;
        }

        if (my_x > 0) {
            int west_neighbor = get_rank_3d(comm, my_x - 1, my_y, my_z);
            MPI_Send(&current_sum, 1, MPI_INT, west_neighbor, 10, comm);
            message_count++;  
        }
        
        if (my_x == 0) {
            *global_result = current_sum;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim = round(cbrt(size));
    if (dim * dim * dim != size) {
        if(rank==0) std::cerr << "Error: Process count must be a perfect cube (e.g., 8, 27, 64)." << std::endl;
        MPI_Finalize(); return 1;
    }
    int dims[3] = {dim, dim, dim};
    int periods[3] = {0, 0, 0};
    int reorder = 1;
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cart_comm);

    int my_val = 1;
    int final_sum = 0;

    MPI_Barrier(cart_comm);
    
    double t1 = MPI_Wtime();
    
    reduce_xyz_pipeline(&my_val, &final_sum, 0, cart_comm); 
    
    double t2 = MPI_Wtime();
    double tmax;
    MPI_Reduce(&t2, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    t2 = tmax;

    int total_messages = 0;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        printf("3D X-Y-Z Pipeline Reduce (%dx%dx%d) Completed.\n", dim, dim, dim);
        printf("Time Taken:   %f\n", t2 - t1);
        printf("Total messages sent: %d\n", total_messages);
        
        if (final_sum != size) printf("FAILED!\n");
        else printf("SUCCESS!\n");
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
