#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

int message_count = 0;  

int get_rank_3d(MPI_Comm comm, int x, int y, int z) {
    int coords[3] = {x, y, z};
    int rank;
    MPI_Cart_rank(comm, coords, &rank);
    return rank;
}

void bcast_xyz_pipeline(int* data, int count, MPI_Datatype datatype, MPI_Comm comm, int dims[3]) {
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
            message_count++;  
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
            message_count++;  
        }
    }

    if (my_z > 0) {
        int source = get_rank_3d(comm, my_x, my_y, my_z - 1);
        MPI_Recv(data, count, datatype, source, 3, comm, &status);
    }

    if (my_z < dim_z - 1) {
        int dest = get_rank_3d(comm, my_x, my_y, my_z + 1);
        MPI_Send(data, count, datatype, dest, 3, comm);
        message_count++;  
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

    int bcast_val = 0;
    int expected_val = 555; 

    if (rank == 0) {
        bcast_val = expected_val;
    }

    double t1 = MPI_Wtime();
    
    bcast_xyz_pipeline(&bcast_val, 1, MPI_INT, cart_comm, dims); 
    
    double t2 = MPI_Wtime();
    double max_time;
    MPI_Reduce(&t2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    t2 = max_time;

    int local_error = (bcast_val != expected_val) ? 1 : 0;
    int total_errors = 0;
    MPI_Reduce(&local_error, &total_errors, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    int total_messages = 0;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        printf("X-Y-Z Pipeline Broadcast (%dx%dx%d) Completed.\n", dim, dim, dim);
        printf("Time Taken:     %f\n", t2 - t1);
        printf("Total messages sent: %d\n", total_messages);
        
        if (total_errors > 0) {
            printf("FAILED! %d nodes did not receive the correct data.\n", total_errors);
        } else {
            printf("SUCCESS! All nodes verified.\n");
        }
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
