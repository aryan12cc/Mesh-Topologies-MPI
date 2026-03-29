#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

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
            MPI_Send(buf, count, type, next_rank, 99, cart);
            message_count++;
        } else if (rank == next_rank) {
            MPI_Recv(buf, count, type, walker_rank, 99, cart, MPI_STATUS_IGNORE);
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
            MPI_Send(buf, count, type, next_rank, 99, cart);
            message_count++;
        } else if (rank == next_rank) {
            MPI_Recv(buf, count, type, walker_rank, 99, cart, MPI_STATUS_IGNORE);
        }

        walker[0] += dir;
    }
}

void bcast_local_subgrid(int* data, int count, MPI_Comm comm, 
                         int my_r, int my_c, 
                         int block_r_start, int block_c_start, 
                         int block_h, int block_w) {
    
    int rel_r = my_r - block_r_start; 
    int rel_c = my_c - block_c_start; 

    MPI_Status status;

    if (rel_r == 0) {
        
        if (rel_c > 0) {
            int west = get_rank(comm, my_r, my_c - 1);
            MPI_Recv(data, count, MPI_INT, west, 0, comm, &status);
        }

        if (rel_c < block_w - 1) {
            int east = get_rank(comm, my_r, my_c + 1);
            MPI_Send(data, count, MPI_INT, east, 0, comm);
            message_count++;
        }
    }

    if (rel_r == 0) {
        if (block_h > 1) {
            int south = get_rank(comm, my_r + 1, my_c);
            MPI_Send(data, count, MPI_INT, south, 1, comm);
            message_count++;
        }
    } else {
        
        int north = get_rank(comm, my_r - 1, my_c);
        MPI_Recv(data, count, MPI_INT, north, 1, comm, &status);

        if (rel_r < block_h - 1) {
            int south = get_rank(comm, my_r + 1, my_c);
            MPI_Send(data, count, MPI_INT, south, 1, comm);
            message_count++;
        }
    }
}

void bcast_hierarchical(int* data, int count, MPI_Comm comm, 
                        int my_r, int my_c, 
                        int grid_dim, int block_dim) {
    
    int my_rank;
    MPI_Comm_rank(comm, &my_rank);

    int block_idx_r = my_r / block_dim; 
    int block_idx_c = my_c / block_dim;

    int leader_r = block_idx_r * block_dim;
    int leader_c = block_idx_c * block_dim;
    int leader_rank = get_rank(comm, leader_r, leader_c);

    bool is_leader = (my_rank == leader_rank);
    bool is_global_root = (my_rank == 0); 

    int global_root_coords[2] = {0, 0};
    
    int num_blocks_across = grid_dim / block_dim;
    
    for (int br = 0; br < num_blocks_across; br++) {
        for (int bc = 0; bc < num_blocks_across; bc++) {
            
            if (br == 0 && bc == 0) continue;

            int dest_coords[2] = {br * block_dim, bc * block_dim};

            cartesian_send(data, count, MPI_INT, global_root_coords, dest_coords, comm);
        }
    }

    MPI_Barrier(comm); 

    bcast_local_subgrid(data, count, comm, 
                        my_r, my_c, 
                        leader_r, leader_c,   
                        block_dim, block_dim 
                       );
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim = sqrt(size);
    if (dim * dim != size) {
        if(rank==0) std::cerr << "Error: Need square grid (e.g. 16, 64)." << std::endl;
        MPI_Finalize(); return 1;
    }

    int block_dim = sqrt(dim); 
    
    if (block_dim * block_dim != dim) {
        if(rank==0) std::cerr << "Error: Grid dimension must be a perfect square for this block strategy." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int dims[2] = {dim, dim};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    int my_coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, my_coords);

    int val = 0;
    if (rank == 0) val = 1234;

    MPI_Barrier(cart_comm);

    double t1 = MPI_Wtime();
    bcast_hierarchical(&val, 1, cart_comm, my_coords[0], my_coords[1], dim, block_dim);
    double t2 = MPI_Wtime();

    double max_t2;
    MPI_Reduce(&t2, &max_t2, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    int total_messages = message_count;;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    if (val != 1234) printf("Rank %d Failed!\n", rank);
    
    if (rank == 0) {
        printf("Hierarchical Broadcast (Grid %dx%d, Blocks %dx%d) Time: %f\n", 
               dim, dim, block_dim, block_dim, max_t2 - t1);
        printf("Total messages sent: %d\n", total_messages);
    }

    MPI_Finalize();
    return 0;
}
