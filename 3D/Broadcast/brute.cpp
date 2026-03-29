#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;

int message_count = 0;  

void cartesian_send_3d(void *buf, int count, MPI_Datatype type,
                       int src_coords[3], int dst_coords[3],
                       MPI_Comm cart)
{
    int rank;
    MPI_Comm_rank(cart, &rank);

    int my_coords[3];
    MPI_Cart_coords(cart, rank, 3, my_coords);

    int walker[3] = {src_coords[0], src_coords[1], src_coords[2]};

    while (walker[0] != dst_coords[0]) {
        int dir = (dst_coords[0] > walker[0]) ? +1 : -1;

        int next_walker[3] = {walker[0] + dir, walker[1], walker[2]};

        int next_rank, walker_rank;
        MPI_Cart_rank(cart, next_walker, &next_rank);
        MPI_Cart_rank(cart, walker, &walker_rank);

        if (rank == walker_rank) {
            MPI_Send(buf, count, type, next_rank, 0, cart);
            message_count++;  
        } else if (rank == next_rank) {
            MPI_Recv(buf, count, type, walker_rank, 0, cart, MPI_STATUS_IGNORE);
        }

        walker[0] += dir;   
    }

    while (walker[1] != dst_coords[1]) {
        int dir = (dst_coords[1] > walker[1]) ? +1 : -1;

        int next_walker[3] = {walker[0], walker[1] + dir, walker[2]};

        int next_rank, walker_rank;
        MPI_Cart_rank(cart, next_walker, &next_rank);
        MPI_Cart_rank(cart, walker, &walker_rank);

        if (rank == walker_rank) {
            MPI_Send(buf, count, type, next_rank, 0, cart);
            message_count++;  
        } else if (rank == next_rank) {
            MPI_Recv(buf, count, type, walker_rank, 0, cart, MPI_STATUS_IGNORE);
        }

        walker[1] += dir;
    }

    while (walker[2] != dst_coords[2]) {
        int dir = (dst_coords[2] > walker[2]) ? +1 : -1;

        int next_walker[3] = {walker[0], walker[1], walker[2] + dir};

        int next_rank, walker_rank;
        MPI_Cart_rank(cart, next_walker, &next_rank);
        MPI_Cart_rank(cart, walker, &walker_rank);

        if (rank == walker_rank) {
            MPI_Send(buf, count, type, next_rank, 0, cart);
            message_count++;  
        } else if (rank == next_rank) {
            MPI_Recv(buf, count, type, walker_rank, 0, cart, MPI_STATUS_IGNORE);
        }

        walker[2] += dir;
    }
}

void bcast_brute_force_3d(int* data, int count, MPI_Datatype datatype, int root_rank, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int root_coords[3];
    MPI_Cart_coords(comm, root_rank, 3, root_coords);

    for (int dest = 0; dest < size; dest++) {
        if (dest == root_rank) continue; 

        int dest_coords[3];
        MPI_Cart_coords(comm, dest, 3, dest_coords);

        int temp_val = 0;

        if (rank == root_rank) {
            temp_val = *data;
        }

        cartesian_send_3d(&temp_val, count, datatype, root_coords, dest_coords, comm);

        if (rank == dest) {
            *data = temp_val;
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
        if(rank==0) cerr << "Error: Process count must be a perfect cube (e.g., 8, 27, 64)." << endl;
        MPI_Finalize(); return 1;
    }

    int dims[3] = {dim, dim, dim};
    int periods[3] = {0, 0, 0};
    int reorder = 1;
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cart_comm);

    int bcast_val = 0;
    int expected_val = 777; 

    if (rank == 0) {
        bcast_val = expected_val;
    }

    MPI_Barrier(cart_comm);

    auto t1 = chrono::high_resolution_clock::now();
    
    bcast_brute_force_3d(&bcast_val, 1, MPI_INT, 0, cart_comm); 
    
    auto t2_local = chrono::high_resolution_clock::now();

    double t2_seconds = chrono::duration<double>(t2_local.time_since_epoch()).count();
    double t2_seconds_max = 0.0;
    MPI_Reduce(&t2_seconds, &t2_seconds_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    int local_error = (bcast_val != expected_val) ? 1 : 0;
    int total_errors = 0;
    
    MPI_Reduce(&local_error, &total_errors, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    int total_messages = 0;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        double t1_seconds = chrono::duration<double>(t1.time_since_epoch()).count();
        double elapsed = t2_seconds_max - t1_seconds;

        printf("3D Brute Force Broadcast (Manual Hop) (%dx%dx%d) Completed.\n", dim, dim, dim);
        printf("Time Taken:     %f\n", elapsed);
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
