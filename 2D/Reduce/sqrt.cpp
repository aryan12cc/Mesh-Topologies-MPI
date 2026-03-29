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

void reduce_hierarchical(int* local_val, int* global_result, int root_rank, MPI_Comm comm, int grid_dim, int block_dim) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int coords[2];
    MPI_Cart_coords(comm, rank, 2, coords);
    int my_r = coords[0];
    int my_c = coords[1];

    int block_start_r = (my_r / block_dim) * block_dim;
    int block_start_c = (my_c / block_dim) * block_dim;
    
    int rel_r = my_r - block_start_r;
    int rel_c = my_c - block_start_c;

    int current_sum = *local_val;
    MPI_Status status;

    if (rel_r < block_dim - 1 && (my_r + 1) < grid_dim) {
        int south_neighbor = get_rank(comm, my_r + 1, my_c);
        int recv_val;
        MPI_Recv(&recv_val, 1, MPI_INT, south_neighbor, 10, comm, &status); 
        current_sum += recv_val;
    }

    if (rel_r > 0) {
        int north_neighbor = get_rank(comm, my_r - 1, my_c);
        MPI_Send(&current_sum, 1, MPI_INT, north_neighbor, 10, comm); 
        message_count++;
    }

    if (rel_r == 0) {
        
        if (rel_c < block_dim - 1 && (my_c + 1) < grid_dim) {
            int east_neighbor = get_rank(comm, my_r, my_c + 1);
            int recv_val;
            MPI_Recv(&recv_val, 1, MPI_INT, east_neighbor, 20, comm, &status); 
            current_sum += recv_val;
        }

        if (rel_c > 0) {
            int west_neighbor = get_rank(comm, my_r, my_c - 1);
            MPI_Send(&current_sum, 1, MPI_INT, west_neighbor, 20, comm); 
            message_count++;
        }
    }

    bool is_leader = (rel_r == 0 && rel_c == 0);
    bool is_global_root = (rank == root_rank); 
    int global_root_coords[2] = {0, 0};

    int num_blocks_across = (grid_dim + block_dim - 1) / block_dim; 
    
    for (int br = 0; br < num_blocks_across; br++) {
        for (int bc = 0; bc < num_blocks_across; bc++) {
            
            if (br == 0 && bc == 0) continue;

            int leader_r = br * block_dim;
            int leader_c = bc * block_dim;
            
            if (leader_r >= grid_dim || leader_c >= grid_dim) continue;

            int src_coords[2] = {leader_r, leader_c};
            int block_val = 0;
            
            if (is_leader && my_r == leader_r && my_c == leader_c) {
                block_val = current_sum;
            }
            
            cartesian_send(&block_val, 1, MPI_INT, src_coords, global_root_coords, comm);
            
            if (is_global_root && !(br == 0 && bc == 0)) {
                current_sum += block_val;
            }
        }
    }
    
    if (is_global_root) {
        *global_result = current_sum;
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
    
    int block_dim = dim / 2; 
    if (block_dim < 1) block_dim = 1;

    int dims[2] = {dim, dim};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int my_val = 1;
    int final_sum = 0;

    double t1 = MPI_Wtime();
    
    reduce_hierarchical(&my_val, &final_sum, 0, cart_comm, dim, block_dim);
    
    double t2 = MPI_Wtime();

    double max_time;
    MPI_Reduce(&t2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    t2 = max_time;

    MPI_Barrier(cart_comm);
    int total_messages = 0;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        printf("Hierarchical Reduce (Block Size %dx%d) Completed.\n", block_dim, block_dim);
        printf("Time Taken:   %f\n", t2 - t1);
        printf("Total messages sent: %d\n", total_messages);
        
        if (final_sum != size) printf("FAILED!\n");
    }

    MPI_Finalize();
    return 0;
}
