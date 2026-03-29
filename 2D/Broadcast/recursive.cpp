#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

int message_cnt = 0;

int get_rank_from_coords(MPI_Comm comm, int row, int col) {
    int coords[2] = {row, col};
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
            message_cnt++;
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
            message_cnt++;
        } else if (rank == next_rank) {
            MPI_Recv(buf, count, type, walker_rank, 0, cart, MPI_STATUS_IGNORE);
        }

        walker[0] += dir;
    }
}

void bcast_recursive_quadrant(int* data, int count, MPI_Datatype datatype, MPI_Comm cart_comm, 
                               int my_row, int my_col, 
                               int r_start, int c_start, 
                               int height, int width) {
    
    if (height <= 1 && width <= 1) {
        return;
    }

    int mid_h = height / 2;
    int mid_w = width / 2;
    
    if (height == 1) mid_h = 0;
    if (width == 1) mid_w = 0;

    int region_root_rank = get_rank_from_coords(cart_comm, r_start, c_start);
    
    int my_rank;
    MPI_Comm_rank(cart_comm, &my_rank);
    MPI_Status status;
    
    int region_root_coords[2] = {r_start, c_start};
    
    if (my_rank == region_root_rank) {
        
        if (mid_w > 0 && c_start + mid_w < c_start + width) {
            int quad1_coords[2] = {r_start, c_start + mid_w};
            cartesian_send(data, count, datatype, region_root_coords, quad1_coords, cart_comm);
        }
        
        if (mid_h > 0 && r_start + mid_h < r_start + height) {
            int quad2_coords[2] = {r_start + mid_h, c_start};
            cartesian_send(data, count, datatype, region_root_coords, quad2_coords, cart_comm);
        }
        
        if (mid_h > 0 && mid_w > 0 && 
            r_start + mid_h < r_start + height && 
            c_start + mid_w < c_start + width) {
            int quad3_coords[2] = {r_start + mid_h, c_start + mid_w};
            cartesian_send(data, count, datatype, region_root_coords, quad3_coords, cart_comm);
        }
    } else {
        
        if (mid_w > 0 && c_start + mid_w < c_start + width) {
            int quad1_coords[2] = {r_start, c_start + mid_w};
            cartesian_send(data, count, datatype, region_root_coords, quad1_coords, cart_comm);
        }
        
        if (mid_h > 0 && r_start + mid_h < r_start + height) {
            int quad2_coords[2] = {r_start + mid_h, c_start};
            cartesian_send(data, count, datatype, region_root_coords, quad2_coords, cart_comm);
        }
        
        if (mid_h > 0 && mid_w > 0 && 
            r_start + mid_h < r_start + height && 
            c_start + mid_w < c_start + width) {
            int quad3_coords[2] = {r_start + mid_h, c_start + mid_w};
            cartesian_send(data, count, datatype, region_root_coords, quad3_coords, cart_comm);
        }
    }
    
    bool in_top_half = (my_row < r_start + mid_h) || mid_h == 0;
    bool in_left_half = (my_col < c_start + mid_w) || mid_w == 0;
    
    int my_quadrant = -1;
    int my_quad_r_start, my_quad_c_start, my_quad_height, my_quad_width;
    
    if (in_top_half && in_left_half) {
        
        my_quadrant = 0;
        my_quad_r_start = r_start;
        my_quad_c_start = c_start;
        my_quad_height = (mid_h > 0) ? mid_h : 1;
        my_quad_width = (mid_w > 0) ? mid_w : 1;
        
    } else if (in_top_half && !in_left_half) {
        
        my_quadrant = 1;
        my_quad_r_start = r_start;
        my_quad_c_start = c_start + mid_w;
        my_quad_height = (mid_h > 0) ? mid_h : 1;
        my_quad_width = width - mid_w;
        
    } else if (!in_top_half && in_left_half) {
        
        my_quadrant = 2;
        my_quad_r_start = r_start + mid_h;
        my_quad_c_start = c_start;
        my_quad_height = height - mid_h;
        my_quad_width = (mid_w > 0) ? mid_w : 1;
        
    } else {
        
        my_quadrant = 3;
        my_quad_r_start = r_start + mid_h;
        my_quad_c_start = c_start + mid_w;
        my_quad_height = height - mid_h;
        my_quad_width = width - mid_w;
    }
    
    if (my_quad_height > 1 || my_quad_width > 1) {
        bcast_recursive_quadrant(data, count, datatype, cart_comm, 
                                 my_row, my_col, 
                                 my_quad_r_start, my_quad_c_start, 
                                 my_quad_height, my_quad_width);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int dim_size = sqrt(world_size);
    if (dim_size * dim_size != world_size) {
        if (world_rank == 0) cerr << "Error: Process count must be a square (e.g., 4, 9, 16)." << std::endl;
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
        message = 555;
        
    }

    MPI_Barrier(cart_comm);

    double start = MPI_Wtime();
    bcast_recursive_quadrant(&message, 1, MPI_INT, cart_comm, 
                            coords[0], coords[1], 
                            0, 0, 
                            dim_size, dim_size);
    double end = MPI_Wtime();

    double max_end;
    MPI_Reduce(&end, &max_end, 1, MPI_DOUBLE, MPI_MAX, root_rank, cart_comm);\

    int total_messages = 0;
    MPI_Reduce(&message_cnt, &total_messages, 1, MPI_INT, MPI_SUM, root_rank, cart_comm);

    if (message == 555) {
        
        if (cart_rank % dim_size == 0 || 1) { 
            
        }
    } else {
        printf("ERROR: Node (Coords %d,%d) did not receive data!\n", coords[0], coords[1]);
    }

    if (cart_rank == root_rank) {
        printf("%d x %d grid:\n", dim_size, dim_size); 
        printf("Time taken: %f seconds\n", max_end - start);
        printf("Total messages sent: %d\n", total_messages);
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
