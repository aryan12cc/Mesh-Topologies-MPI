#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
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

void bcast_brute_force(int* data, int count, MPI_Datatype datatype, int root_rank, MPI_Comm cart_comm) {
    int rank, size;
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Comm_size(cart_comm, &size);

    int root_coords[2];
    MPI_Cart_coords(cart_comm, root_rank, 2, root_coords);

    for (int dest = 0; dest < size; dest++) {
        if (dest != root_rank) {
            
            int dest_coords[2];
            MPI_Cart_coords(cart_comm, dest, 2, dest_coords);

            cartesian_send(data, count, datatype, root_coords, dest_coords, cart_comm);
            
            MPI_Barrier(cart_comm);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int dim_size = sqrt(world_size);
    if (dim_size * dim_size != world_size) {
        if (world_rank == 0) cerr << "Error: Process count must be a square (e.g., 4, 9, 16)." << endl;
        MPI_Finalize();
        return 1;
    }

    int dims[2] = {dim_size, dim_size};
    int periods[2] = {0, 0};
    int reorder = 1;
    MPI_Comm cart_comm;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    int cart_rank;
    int coords[2];
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    int root_coords[2] = {0, 0};
    int root_rank;
    MPI_Cart_rank(cart_comm, root_coords, &root_rank);

    int message = 0;
    if (cart_rank == root_rank) {
        message = 42;
    }

    MPI_Barrier(cart_comm);

    double start = MPI_Wtime();
    bcast_brute_force(&message, 1, MPI_INT, root_rank, cart_comm);
    double end = MPI_Wtime();

    double max_end;
    MPI_Reduce(&end, &max_end, 1, MPI_DOUBLE, MPI_MAX, root_rank, cart_comm);
    end = max_end;

    if (message != 42) {
        printf("ERROR: Node (Coords %d,%d) did not receive data!\n", coords[0], coords[1]);
    }

    int total_messages = 0;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, root_rank, cart_comm);

    if (cart_rank == root_rank) {
        printf("%d x %d grid:\n", dim_size, dim_size); 
        printf("Time taken: %f seconds\n", end - start);
        printf("Total messages sent: %d\n", total_messages);
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
