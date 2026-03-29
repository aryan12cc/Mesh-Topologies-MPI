#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

int message_count = 0;  

int get_rank_3d(MPI_Comm comm, int x, int y, int z)
{
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

    while (walker[0] != dst_coords[0])
    {
        int dir = (dst_coords[0] > walker[0]) ? +1 : -1;
        int next_walker[3] = {walker[0] + dir, walker[1], walker[2]};
        int next_rank, walker_rank;
        MPI_Cart_rank(cart, next_walker, &next_rank);
        MPI_Cart_rank(cart, walker, &walker_rank);

        if (rank == walker_rank)
        {
            MPI_Send(buf, count, type, next_rank, 0, cart);
            message_count++;  
        }
        else if (rank == next_rank)
        {
            MPI_Recv(buf, count, type, walker_rank, 0, cart, MPI_STATUS_IGNORE);
        }
        walker[0] += dir;
    }

    while (walker[1] != dst_coords[1])
    {
        int dir = (dst_coords[1] > walker[1]) ? +1 : -1;
        int next_walker[3] = {walker[0], walker[1] + dir, walker[2]};
        int next_rank, walker_rank;
        MPI_Cart_rank(cart, next_walker, &next_rank);
        MPI_Cart_rank(cart, walker, &walker_rank);

        if (rank == walker_rank)
        {
            MPI_Send(buf, count, type, next_rank, 0, cart);
            message_count++;  
        }
        else if (rank == next_rank)
        {
            MPI_Recv(buf, count, type, walker_rank, 0, cart, MPI_STATUS_IGNORE);
        }
        walker[1] += dir;
    }

    while (walker[2] != dst_coords[2])
    {
        int dir = (dst_coords[2] > walker[2]) ? +1 : -1;
        int next_walker[3] = {walker[0], walker[1], walker[2] + dir};
        int next_rank, walker_rank;
        MPI_Cart_rank(cart, next_walker, &next_rank);
        MPI_Cart_rank(cart, walker, &walker_rank);

        if (rank == walker_rank)
        {
            MPI_Send(buf, count, type, next_rank, 0, cart);
            message_count++;  
        }
        else if (rank == next_rank)
        {
            MPI_Recv(buf, count, type, walker_rank, 0, cart, MPI_STATUS_IGNORE);
        }
        walker[2] += dir;
    }
}

void bcast_local_subgrid_3d(int *data, int count, MPI_Comm comm, int my_x, int my_y, int my_z,
                            int block_x_start, int block_y_start, int block_z_start, int block_dim)
{

    int rel_x = my_x - block_x_start;
    int rel_y = my_y - block_y_start;
    int rel_z = my_z - block_z_start;

    MPI_Status status;

    if (rel_y == 0 && rel_z == 0)
    {
        
        if (rel_x > 0)
        {
            int west = get_rank_3d(comm, my_x - 1, my_y, my_z);
            MPI_Recv(data, count, MPI_INT, west, 1, comm, &status);
        }

        if (rel_x < block_dim - 1)
        {
            int east = get_rank_3d(comm, my_x + 1, my_y, my_z);
            MPI_Send(data, count, MPI_INT, east, 1, comm);
            message_count++;  
        }
    }

    if (rel_z == 0)
    {
        
        if (rel_y > 0)
        {
            int north = get_rank_3d(comm, my_x, my_y - 1, my_z);
            MPI_Recv(data, count, MPI_INT, north, 2, comm, &status);
        }

        if (rel_y < block_dim - 1)
        {
            int south = get_rank_3d(comm, my_x, my_y + 1, my_z);
            MPI_Send(data, count, MPI_INT, south, 2, comm);
            message_count++;  
        }
    }

    if (rel_z > 0)
    {
        int front = get_rank_3d(comm, my_x, my_y, my_z - 1);
        MPI_Recv(data, count, MPI_INT, front, 3, comm, &status);
    }

    if (rel_z < block_dim - 1)
    {
        int back = get_rank_3d(comm, my_x, my_y, my_z + 1);
        MPI_Send(data, count, MPI_INT, back, 3, comm);
        message_count++;  
    }
}

void bcast_hierarchical_3d(int *data, int count, MPI_Comm comm,
                           int my_x, int my_y, int my_z,
                           int grid_dim, int block_dim)
{

    int my_rank;
    MPI_Comm_rank(comm, &my_rank);

    int block_idx_x = my_x / block_dim;
    int block_idx_y = my_y / block_dim;
    int block_idx_z = my_z / block_dim;

    int leader_x = block_idx_x * block_dim;
    int leader_y = block_idx_y * block_dim;
    int leader_z = block_idx_z * block_dim;
    int leader_rank = get_rank_3d(comm, leader_x, leader_y, leader_z);

    bool is_leader = (my_rank == leader_rank);
    bool is_global_root = (my_rank == 0); 

    int global_root_coords[3] = {0, 0, 0};
    int num_blocks = grid_dim / block_dim;

    for (int bx = 0; bx < num_blocks; bx++)
    {
        for (int by = 0; by < num_blocks; by++)
        {
            for (int bz = 0; bz < num_blocks; bz++)
            {
                
                if (bx == 0 && by == 0 && bz == 0)
                    continue;

                int dest_coords[3] = {bx * block_dim, by * block_dim, bz * block_dim};

                cartesian_send_3d(data, count, MPI_INT, global_root_coords, dest_coords, comm);
            }
        }
    }

    bcast_local_subgrid_3d(data, count, comm,
                           my_x, my_y, my_z,
                           leader_x, leader_y, leader_z,
                           block_dim);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim = round(cbrt(size));
    if (dim * dim * dim != size)
    {
        if (rank == 0)
            std::cerr << "Error: Process count must be a perfect cube (e.g. 64, 512, 4096)." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int block_dim = sqrt(dim);

    if (block_dim * block_dim != dim)
    {
        if (rank == 0)
            cerr << "Error: Grid dimension (" << dim << ") must be a perfect square for this block strategy." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int dims[3] = {dim, dim, dim};
    int periods[3] = {0, 0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);

    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    int val = 0;
    if (rank == 0)
        val = 9998;

    double t1 = MPI_Wtime();

    bcast_hierarchical_3d(&val, 1, cart_comm, coords[0], coords[1], coords[2], dim, block_dim);

    double t2 = MPI_Wtime();
    double max_time;
    MPI_Reduce(&t2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    t2 = max_time;

    int local_error = (val != 9998) ? 1 : 0;
    int total_errors = 0;
    MPI_Reduce(&local_error, &total_errors, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    int total_messages = 0;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    if (rank == 0)
    {
        printf("3D Hierarchical Broadcast (Grid %dx%dx%d, Blocks %dx%dx%d) Completed.\n",
               dim, dim, dim, block_dim, block_dim, block_dim);
        printf("Time Taken: %f\n", t2 - t1);
        printf("Total messages sent: %d\n", total_messages);

        if (total_errors > 0)
            printf("FAILED! %d nodes had errors.\n", total_errors);
        else
            printf("SUCCESS! All nodes verified.\n");
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
