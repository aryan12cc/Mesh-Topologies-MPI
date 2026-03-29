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

void bcast_recursive_octant_3d(int* data, int count, MPI_Datatype datatype, MPI_Comm cart_comm,
                                int my_x, int my_y, int my_z,
                                int x_start, int y_start, int z_start,
                                int width, int height, int depth) {
    
    if (width <= 1 && height <= 1 && depth <= 1) {
        return;
    }

    int mid_w = width / 2;
    int mid_h = height / 2;
    int mid_d = depth / 2;
    
    if (width == 1) mid_w = 0;
    if (height == 1) mid_h = 0;
    if (depth == 1) mid_d = 0;

    int region_root_coords[3] = {x_start, y_start, z_start};
    
    if (mid_w > 0) {
        int oct1_coords[3] = {x_start + mid_w, y_start, z_start};
        cartesian_send_3d(data, count, datatype, region_root_coords, oct1_coords, cart_comm);
    }
    
    if (mid_h > 0) {
        int oct2_coords[3] = {x_start, y_start + mid_h, z_start};
        cartesian_send_3d(data, count, datatype, region_root_coords, oct2_coords, cart_comm);
    }
    
    if (mid_w > 0 && mid_h > 0) {
        int oct3_coords[3] = {x_start + mid_w, y_start + mid_h, z_start};
        cartesian_send_3d(data, count, datatype, region_root_coords, oct3_coords, cart_comm);
    }
    
    if (mid_d > 0) {
        int oct4_coords[3] = {x_start, y_start, z_start + mid_d};
        cartesian_send_3d(data, count, datatype, region_root_coords, oct4_coords, cart_comm);
    }
    
    if (mid_w > 0 && mid_d > 0) {
        int oct5_coords[3] = {x_start + mid_w, y_start, z_start + mid_d};
        cartesian_send_3d(data, count, datatype, region_root_coords, oct5_coords, cart_comm);
    }
    
    if (mid_h > 0 && mid_d > 0) {
        int oct6_coords[3] = {x_start, y_start + mid_h, z_start + mid_d};
        cartesian_send_3d(data, count, datatype, region_root_coords, oct6_coords, cart_comm);
    }
    
    if (mid_w > 0 && mid_h > 0 && mid_d > 0) {
        int oct7_coords[3] = {x_start + mid_w, y_start + mid_h, z_start + mid_d};
        cartesian_send_3d(data, count, datatype, region_root_coords, oct7_coords, cart_comm);
    }
    
    bool in_left = (my_x < x_start + mid_w) || (mid_w == 0);
    bool in_top = (my_y < y_start + mid_h) || (mid_h == 0);
    bool in_front = (my_z < z_start + mid_d) || (mid_d == 0);
    
    int my_oct_x_start, my_oct_y_start, my_oct_z_start;
    int my_oct_width, my_oct_height, my_oct_depth;
    
    if (in_left) {
        my_oct_x_start = x_start;
        my_oct_width = (mid_w > 0) ? mid_w : width;
    } else {
        my_oct_x_start = x_start + mid_w;
        my_oct_width = width - mid_w;
    }
    
    if (in_top) {
        my_oct_y_start = y_start;
        my_oct_height = (mid_h > 0) ? mid_h : height;
    } else {
        my_oct_y_start = y_start + mid_h;
        my_oct_height = height - mid_h;
    }
    
    if (in_front) {
        my_oct_z_start = z_start;
        my_oct_depth = (mid_d > 0) ? mid_d : depth;
    } else {
        my_oct_z_start = z_start + mid_d;
        my_oct_depth = depth - mid_d;
    }
    
    bcast_recursive_octant_3d(data, count, datatype, cart_comm,
                              my_x, my_y, my_z,
                              my_oct_x_start, my_oct_y_start, my_oct_z_start,
                              my_oct_width, my_oct_height, my_oct_depth);
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

    int message = 0;
    int expected_val = 777;
    
    if (cart_rank == root_rank) {
        message = expected_val;
    }

    double start = MPI_Wtime();
    bcast_recursive_octant_3d(&message, 1, MPI_INT, cart_comm,
                              coords[0], coords[1], coords[2],
                              0, 0, 0,
                              dim_size, dim_size, dim_size);
    double end = MPI_Wtime();

    double max_time;
    MPI_Reduce(&end, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    end = max_time;
    int local_error = (message != expected_val) ? 1 : 0;
    int total_errors = 0;
    
    MPI_Reduce(&local_error, &total_errors, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    int total_messages = 0;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    if (cart_rank == root_rank) {
        printf("3D Recursive Octant Broadcast (%dx%dx%d) Completed.\n", dim_size, dim_size, dim_size);
        printf("Time Taken:     %f seconds\n", end - start);
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
