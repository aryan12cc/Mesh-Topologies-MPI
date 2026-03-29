#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

int message_cnt = 0;

int get_rank(MPI_Comm comm, int r, int c) {
    int coords[2] = {r, c};
    int rank;
    MPI_Cart_rank(comm, coords, &rank);
    return rank;
}

void reduce_row_col(int* local_val, int* global_result, int root_rank, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int dims[2], periods[2], coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);
    int my_r = coords[0];
    int my_c = coords[1];
    int rows = dims[0];
    int cols = dims[1];

    int current_sum = *local_val;
    MPI_Status status;

    if (my_r < rows - 1) {
        int south_neighbor = get_rank(comm, my_r + 1, my_c);
        int received_val;
        
        MPI_Recv(&received_val, 1, MPI_INT, south_neighbor, 1, comm, &status);
        current_sum += received_val;
    }

    if (my_r > 0) {
        int north_neighbor = get_rank(comm, my_r - 1, my_c);
        
        MPI_Send(&current_sum, 1, MPI_INT, north_neighbor, 1, comm);
        message_cnt++;
    }

    if (my_r == 0) {
        
        if (my_c < cols - 1) {
            int east_neighbor = get_rank(comm, my_r, my_c + 1);
            int received_val;
            
            MPI_Recv(&received_val, 1, MPI_INT, east_neighbor, 0, comm, &status);
            current_sum += received_val;
        }

        if (my_c > 0) {
            int west_neighbor = get_rank(comm, my_r, my_c - 1);
            
            MPI_Send(&current_sum, 1, MPI_INT, west_neighbor, 0, comm);
            message_cnt++;
        }
        
        if (my_c == 0) {
            *global_result = current_sum;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim = sqrt(size);
    if (dim * dim != size) {
        if(rank==0) std::cerr << "Error: Square grid required." << std::endl;
        MPI_Finalize(); return 1;
    }
    int dims[2] = {dim, dim};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int my_val = 1;
    int final_sum = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    reduce_row_col(&my_val, &final_sum, 0, cart_comm); 
    auto t2_local = std::chrono::high_resolution_clock::now();

    int total_messages = 0;
    MPI_Reduce(&message_cnt, &total_messages, 1, MPI_INT, MPI_SUM, 0, cart_comm);
    
    if (rank == 0) {
        double t1_seconds = std::chrono::duration<double>(t1.time_since_epoch()).count();
        double elapsed = std::chrono::duration<double>(t2_local - t1).count();

        printf("Row-Column Pipeline Reduce Completed.\n");
        printf("Time Taken:   %f\n", elapsed);
        printf("Total messages sent: %d\n", total_messages);
        
        if (final_sum != size) printf("FAILED!\n");
    }

    MPI_Finalize();
    return 0;
}
