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

void reduce_recursive_octant_3d(int* local_data, int* result, MPI_Comm cart_comm,
                                 int my_x, int my_y, int my_z,
                                 int x_start, int y_start, int z_start,
                                 int width, int height, int depth) {
    
    if (width <= 1 && height <= 1 && depth <= 1) {
        *result = *local_data;
        return;
    }

    int mid_w = width / 2;
    int mid_h = height / 2;
    int mid_d = depth / 2;
    
    if (width == 1) mid_w = 0;
    if (height == 1) mid_h = 0;
    if (depth == 1) mid_d = 0;

    int region_root_rank = get_rank_3d(cart_comm, x_start, y_start, z_start);
    
    int my_rank;
    MPI_Comm_rank(cart_comm, &my_rank);
    MPI_Status status;
    
    bool in_left_half = (my_x < x_start + mid_w) || mid_w == 0;
    bool in_top_half = (my_y < y_start + mid_h) || mid_h == 0;
    bool in_front_half = (my_z < z_start + mid_d) || mid_d == 0;
    
    int my_octant = -1;
    int my_oct_x_start, my_oct_y_start, my_oct_z_start;
    int my_oct_width, my_oct_height, my_oct_depth;
    int my_oct_result = 0;
    
    if (in_front_half && in_top_half && in_left_half) {
        
        my_octant = 0;
        my_oct_x_start = x_start;
        my_oct_y_start = y_start;
        my_oct_z_start = z_start;
        my_oct_width = (mid_w > 0) ? mid_w : 1;
        my_oct_height = (mid_h > 0) ? mid_h : 1;
        my_oct_depth = (mid_d > 0) ? mid_d : 1;
        
    } else if (in_front_half && in_top_half && !in_left_half) {
        
        my_octant = 1;
        my_oct_x_start = x_start + mid_w;
        my_oct_y_start = y_start;
        my_oct_z_start = z_start;
        my_oct_width = width - mid_w;
        my_oct_height = (mid_h > 0) ? mid_h : 1;
        my_oct_depth = (mid_d > 0) ? mid_d : 1;
        
    } else if (in_front_half && !in_top_half && in_left_half) {
        
        my_octant = 2;
        my_oct_x_start = x_start;
        my_oct_y_start = y_start + mid_h;
        my_oct_z_start = z_start;
        my_oct_width = (mid_w > 0) ? mid_w : 1;
        my_oct_height = height - mid_h;
        my_oct_depth = (mid_d > 0) ? mid_d : 1;
        
    } else if (in_front_half && !in_top_half && !in_left_half) {
        
        my_octant = 3;
        my_oct_x_start = x_start + mid_w;
        my_oct_y_start = y_start + mid_h;
        my_oct_z_start = z_start;
        my_oct_width = width - mid_w;
        my_oct_height = height - mid_h;
        my_oct_depth = (mid_d > 0) ? mid_d : 1;
        
    } else if (!in_front_half && in_top_half && in_left_half) {
        
        my_octant = 4;
        my_oct_x_start = x_start;
        my_oct_y_start = y_start;
        my_oct_z_start = z_start + mid_d;
        my_oct_width = (mid_w > 0) ? mid_w : 1;
        my_oct_height = (mid_h > 0) ? mid_h : 1;
        my_oct_depth = depth - mid_d;
        
    } else if (!in_front_half && in_top_half && !in_left_half) {
        
        my_octant = 5;
        my_oct_x_start = x_start + mid_w;
        my_oct_y_start = y_start;
        my_oct_z_start = z_start + mid_d;
        my_oct_width = width - mid_w;
        my_oct_height = (mid_h > 0) ? mid_h : 1;
        my_oct_depth = depth - mid_d;
        
    } else if (!in_front_half && !in_top_half && in_left_half) {
        
        my_octant = 6;
        my_oct_x_start = x_start;
        my_oct_y_start = y_start + mid_h;
        my_oct_z_start = z_start + mid_d;
        my_oct_width = (mid_w > 0) ? mid_w : 1;
        my_oct_height = height - mid_h;
        my_oct_depth = depth - mid_d;
        
    } else {
        
        my_octant = 7;
        my_oct_x_start = x_start + mid_w;
        my_oct_y_start = y_start + mid_h;
        my_oct_z_start = z_start + mid_d;
        my_oct_width = width - mid_w;
        my_oct_height = height - mid_h;
        my_oct_depth = depth - mid_d;
    }
    
    if (my_oct_width > 1 || my_oct_height > 1 || my_oct_depth > 1) {
        reduce_recursive_octant_3d(local_data, &my_oct_result, cart_comm,
                                   my_x, my_y, my_z,
                                   my_oct_x_start, my_oct_y_start, my_oct_z_start,
                                   my_oct_width, my_oct_height, my_oct_depth);
    } else {
        
        my_oct_result = *local_data;
    }
    
    bool oct1_exists = (mid_w > 0 && x_start + mid_w < x_start + width);
    bool oct2_exists = (mid_h > 0 && y_start + mid_h < y_start + height);
    bool oct3_exists = (mid_w > 0 && mid_h > 0 && 
                        x_start + mid_w < x_start + width && 
                        y_start + mid_h < y_start + height);
    bool oct4_exists = (mid_d > 0 && z_start + mid_d < z_start + depth);
    bool oct5_exists = (mid_w > 0 && mid_d > 0 && 
                        x_start + mid_w < x_start + width && 
                        z_start + mid_d < z_start + depth);
    bool oct6_exists = (mid_h > 0 && mid_d > 0 && 
                        y_start + mid_h < y_start + height && 
                        z_start + mid_d < z_start + depth);
    bool oct7_exists = (mid_w > 0 && mid_h > 0 && mid_d > 0 && 
                        x_start + mid_w < x_start + width && 
                        y_start + mid_h < y_start + height && 
                        z_start + mid_d < z_start + depth);
    
    int region_root_coords[3] = {x_start, y_start, z_start};
    
    if (oct1_exists) {
        int oct1_coords[3] = {x_start + mid_w, y_start, z_start};
        int recv_val = 0;
        
        if (my_octant == 1 && my_x == my_oct_x_start && my_y == my_oct_y_start && my_z == my_oct_z_start) {
            recv_val = my_oct_result;
        }
        
        cartesian_send_3d(&recv_val, 1, MPI_INT, oct1_coords, region_root_coords, cart_comm);
        
        if (my_rank == region_root_rank) {
            my_oct_result += recv_val;
        }
    }
    
    if (oct2_exists) {
        int oct2_coords[3] = {x_start, y_start + mid_h, z_start};
        int recv_val = 0;
        
        if (my_octant == 2 && my_x == my_oct_x_start && my_y == my_oct_y_start && my_z == my_oct_z_start) {
            recv_val = my_oct_result;
        }
        
        cartesian_send_3d(&recv_val, 1, MPI_INT, oct2_coords, region_root_coords, cart_comm);
        
        if (my_rank == region_root_rank) {
            my_oct_result += recv_val;
        }
    }
    
    if (oct3_exists) {
        int oct3_coords[3] = {x_start + mid_w, y_start + mid_h, z_start};
        int recv_val = 0;
        
        if (my_octant == 3 && my_x == my_oct_x_start && my_y == my_oct_y_start && my_z == my_oct_z_start) {
            recv_val = my_oct_result;
        }
        
        cartesian_send_3d(&recv_val, 1, MPI_INT, oct3_coords, region_root_coords, cart_comm);
        
        if (my_rank == region_root_rank) {
            my_oct_result += recv_val;
        }
    }
    
    if (oct4_exists) {
        int oct4_coords[3] = {x_start, y_start, z_start + mid_d};
        int recv_val = 0;
        
        if (my_octant == 4 && my_x == my_oct_x_start && my_y == my_oct_y_start && my_z == my_oct_z_start) {
            recv_val = my_oct_result;
        }
        
        cartesian_send_3d(&recv_val, 1, MPI_INT, oct4_coords, region_root_coords, cart_comm);
        
        if (my_rank == region_root_rank) {
            my_oct_result += recv_val;
        }
    }
    
    if (oct5_exists) {
        int oct5_coords[3] = {x_start + mid_w, y_start, z_start + mid_d};
        int recv_val = 0;
        
        if (my_octant == 5 && my_x == my_oct_x_start && my_y == my_oct_y_start && my_z == my_oct_z_start) {
            recv_val = my_oct_result;
        }
        
        cartesian_send_3d(&recv_val, 1, MPI_INT, oct5_coords, region_root_coords, cart_comm);
        
        if (my_rank == region_root_rank) {
            my_oct_result += recv_val;
        }
    }
    
    if (oct6_exists) {
        int oct6_coords[3] = {x_start, y_start + mid_h, z_start + mid_d};
        int recv_val = 0;
        
        if (my_octant == 6 && my_x == my_oct_x_start && my_y == my_oct_y_start && my_z == my_oct_z_start) {
            recv_val = my_oct_result;
        }
        
        cartesian_send_3d(&recv_val, 1, MPI_INT, oct6_coords, region_root_coords, cart_comm);
        
        if (my_rank == region_root_rank) {
            my_oct_result += recv_val;
        }
    }
    
    if (oct7_exists) {
        int oct7_coords[3] = {x_start + mid_w, y_start + mid_h, z_start + mid_d};
        int recv_val = 0;
        
        if (my_octant == 7 && my_x == my_oct_x_start && my_y == my_oct_y_start && my_z == my_oct_z_start) {
            recv_val = my_oct_result;
        }
        
        cartesian_send_3d(&recv_val, 1, MPI_INT, oct7_coords, region_root_coords, cart_comm);
        
        if (my_rank == region_root_rank) {
            my_oct_result += recv_val;
        }
    }
    
    if (my_rank == region_root_rank) {
        *result = my_oct_result;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int dim_size = round(cbrt(world_size));
    if (dim_size * dim_size * dim_size != world_size) {
        if (world_rank == 0) cerr << "Error: Process count must be a perfect cube (e.g., 8, 27, 64)." << endl;
        MPI_Finalize();
        return 1;
    }

    int dims[3] = {dim_size, dim_size, dim_size};
    int periods[3] = {0, 0, 0}; 
    int reorder = 1;
    MPI_Comm cart_comm;

    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cart_comm);

    int cart_rank;
    int coords[3];
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 3, coords);

    int root_coords[3] = {0, 0, 0};
    int root_rank;
    MPI_Cart_rank(cart_comm, root_coords, &root_rank);

    int local_value = 1;
    int final_sum = 0;

    double start = MPI_Wtime();
    reduce_recursive_octant_3d(&local_value, &final_sum, cart_comm,
                               coords[0], coords[1], coords[2],
                               0, 0, 0,
                               dim_size, dim_size, dim_size);
    double end = MPI_Wtime();
    double tmax;
    MPI_Reduce(&end, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    end = tmax;

    int total_messages = 0;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, root_rank, cart_comm);

    if (cart_rank == root_rank) {
        printf("3D Recursive Octant Reduce (%dx%dx%d) Completed.\n", dim_size, dim_size, dim_size);
        printf("Time Taken:   %f seconds\n", end - start);
        printf("Total messages sent: %d\n", total_messages);
        
        if (final_sum == world_size) {
            printf("SUCCESS!\n");
        } else {
            printf("FAILED!\n");
        }
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
