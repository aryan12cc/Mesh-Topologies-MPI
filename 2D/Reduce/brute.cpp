#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;

int message_count = 0;  

int get_rank(MPI_Comm comm, int r, int c) {
    int coords[2] = {r, c};
    int rank;
    MPI_Cart_rank(comm, coords, &rank);
    return rank;
}

void cartesian_send(void *buf, int count, MPI_Datatype type, 
                    int src_coords[2], int dst_coords[2], 
                    MPI_Comm cart) {
    
    int rank;
    MPI_Comm_rank(cart, &rank);
    
    int my_coords[2];
    MPI_Cart_coords(cart, rank, 2, my_coords);

    int walker[2] = {src_coords[0], src_coords[1]};

    while (walker[1] != dst_coords[1]) {
        int dir = (dst_coords[1] > walker[1]) ? +1 : -1;
        
        int next_walker[2] = {walker[0], walker[1] + dir};
        int next_rank;
        MPI_Cart_rank(cart, next_walker, &next_rank);

        int walker_rank;
        MPI_Cart_rank(cart, walker, &walker_rank);

        if (rank == walker_rank) {
            MPI_Send(buf, count, type, next_rank, 0, cart);
            message_count++;
            
        } else if (rank == next_rank) {
            MPI_Recv(buf, count, type, walker_rank, 0, cart, MPI_STATUS_IGNORE);
        }

        walker[1] += dir;
    }

    while (walker[0] != dst_coords[0]) {
        int dir = (dst_coords[0] > walker[0]) ? +1 : -1;
        
        int next_walker[2] = {walker[0] + dir, walker[1]};
        int next_rank;
        MPI_Cart_rank(cart, next_walker, &next_rank);

        int walker_rank;
        MPI_Cart_rank(cart, walker, &walker_rank);

        if (rank == walker_rank) {
            MPI_Send(buf, count, type, next_rank, 0, cart);
            message_count++;
            
        } else if (rank == next_rank) {
            MPI_Recv(buf, count, type, walker_rank, 0, cart, MPI_STATUS_IGNORE);
        }

        walker[0] += dir;
    }
}

void reduce_brute_force(int* local_val, int* global_result, int root_rank, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int root_coords[2];
    MPI_Cart_coords(comm, root_rank, 2, root_coords);

    int running_sum = 0;

    for (int src = 0; src < size; src++) {
        
        int src_coords[2];
        MPI_Cart_coords(comm, src, 2, src_coords);

        int temp_val = 0;

        if (rank == src) {
            temp_val = *local_val;
        }

        if (src != root_rank) {
            cartesian_send(&temp_val, 1, MPI_INT, src_coords, root_coords, comm);
            
            MPI_Barrier(comm);
        } else {
            
            if (rank == root_rank) {
                temp_val = *local_val;
            }
            MPI_Barrier(comm);
        }

        if (rank == root_rank) {
            running_sum += temp_val;
        }
    }

    if (rank == root_rank) {
        *global_result = running_sum;
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

    MPI_Barrier(cart_comm);

    auto t1 = std::chrono::high_resolution_clock::now();
    
    reduce_brute_force(&my_val, &final_sum, 0, cart_comm); 
    
    auto t2_local = std::chrono::high_resolution_clock::now();

    double t2_seconds = std::chrono::duration<double>(t2_local.time_since_epoch()).count();
    double t2_seconds_max = 0.0;
    MPI_Reduce(&t2_seconds, &t2_seconds_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    int total_messages = 0;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        double t1_seconds = std::chrono::duration<double>(t1.time_since_epoch()).count();
        double elapsed = t2_seconds_max - t1_seconds;

        printf("Manual Hop Brute Force Reduce Completed.\n");
        printf("Time Taken:   %f\n", elapsed);
        printf("Total messages sent: %d\n", total_messages);
        
        if (final_sum != size) printf("FAILED!\n");
        else printf("SUCCESS!\n");
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
