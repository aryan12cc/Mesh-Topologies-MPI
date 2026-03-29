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

void reduce_hierarchical_3d(int* local_val, int* global_result, int root_rank, MPI_Comm comm, int grid_dim, int block_dim) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int coords[3];
    MPI_Cart_coords(comm, rank, 3, coords);
    int my_x = coords[0];
    int my_y = coords[1];
    int my_z = coords[2];

    int block_start_x = (my_x / block_dim) * block_dim;
    int block_start_y = (my_y / block_dim) * block_dim;
    int block_start_z = (my_z / block_dim) * block_dim;
    
    int rel_x = my_x - block_start_x;
    int rel_y = my_y - block_start_y;
    int rel_z = my_z - block_start_z;

    int current_sum = *local_val;
    MPI_Status status;

    if (rel_z < block_dim - 1 && (my_z + 1) < grid_dim) {
        int back_neighbor = get_rank_3d(comm, my_x, my_y, my_z + 1);
        int recv_val;
        MPI_Recv(&recv_val, 1, MPI_INT, back_neighbor, 30, comm, &status); 
        current_sum += recv_val;
    }

    if (rel_z > 0) {
        int front_neighbor = get_rank_3d(comm, my_x, my_y, my_z - 1);
        MPI_Send(&current_sum, 1, MPI_INT, front_neighbor, 30, comm); 
        message_count++;  
    }

    if (rel_z == 0) {
        
        if (rel_y < block_dim - 1 && (my_y + 1) < grid_dim) {
            int south_neighbor = get_rank_3d(comm, my_x, my_y + 1, my_z);
            int recv_val;
            MPI_Recv(&recv_val, 1, MPI_INT, south_neighbor, 20, comm, &status); 
            current_sum += recv_val;
        }

        if (rel_y > 0) {
            int north_neighbor = get_rank_3d(comm, my_x, my_y - 1, my_z);
            MPI_Send(&current_sum, 1, MPI_INT, north_neighbor, 20, comm); 
            message_count++;  
        }
    }

    if (rel_z == 0 && rel_y == 0) {
        
        if (rel_x < block_dim - 1 && (my_x + 1) < grid_dim) {
            int east_neighbor = get_rank_3d(comm, my_x + 1, my_y, my_z);
            int recv_val;
            MPI_Recv(&recv_val, 1, MPI_INT, east_neighbor, 10, comm, &status); 
            current_sum += recv_val;
        }

        if (rel_x > 0) {
            int west_neighbor = get_rank_3d(comm, my_x - 1, my_y, my_z);
            MPI_Send(&current_sum, 1, MPI_INT, west_neighbor, 10, comm); 
            message_count++;  
        }
    }

    bool is_leader = (rel_x == 0 && rel_y == 0 && rel_z == 0);
    bool is_global_root = (rank == root_rank);
    int global_root_coords[3] = {0, 0, 0};

    int num_blocks = (grid_dim + block_dim - 1) / block_dim; 

    for (int bx = 0; bx < num_blocks; bx++) {
        for (int by = 0; by < num_blocks; by++) {
            for (int bz = 0; bz < num_blocks; bz++) {
                
                if (bx == 0 && by == 0 && bz == 0) continue;

                int leader_x = bx * block_dim;
                int leader_y = by * block_dim;
                int leader_z = bz * block_dim;

                if (leader_x >= grid_dim || leader_y >= grid_dim || leader_z >= grid_dim) continue;

                int src_coords[3] = {leader_x, leader_y, leader_z};
                int block_val = 0;
                
                if (is_leader && my_x == leader_x && my_y == leader_y && my_z == leader_z) {
                    block_val = current_sum;
                }
                
                cartesian_send_3d(&block_val, 1, MPI_INT, src_coords, global_root_coords, comm);
                
                if (is_global_root && !(bx == 0 && by == 0 && bz == 0)) {
                    current_sum += block_val;
                }
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

    int dim = round(cbrt(size));
    if (dim * dim * dim != size) {
        if(rank==0) cerr << "Error: Process count must be a perfect cube." << endl;
        MPI_Finalize();
        return 1;
    }

    int block_dim = sqrt(dim); 
    if (block_dim < 1) block_dim = 1;

    int dims[3] = {dim, dim, dim};
    int periods[3] = {0, 0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);

    int my_val = 1;
    int final_sum = 0;

    MPI_Barrier(cart_comm);

    double t1 = MPI_Wtime();
    
    reduce_hierarchical_3d(&my_val, &final_sum, 0, cart_comm, dim, block_dim);
    
    double t2 = MPI_Wtime();
    double tmax;
    MPI_Reduce(&t2, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    t2 = tmax;

    int total_messages = 0;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        printf("3D Hierarchical Reduce (Block Size %dx%dx%d) Completed.\n", block_dim, block_dim, block_dim);
        printf("Time Taken:   %f\n", t2 - t1);
        printf("Total messages sent: %d\n", total_messages);
        
        if (final_sum != size) printf("FAILED!\n");
        else printf("SUCCESS!\n");
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
