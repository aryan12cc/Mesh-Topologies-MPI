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

void bcast_row_col(int* data, int count, MPI_Datatype datatype, MPI_Comm cart_comm, int my_row, int my_col, int dims[2]) {
    int rows = dims[0];
    int cols = dims[1];
    MPI_Status status;

    if (my_row == 0) {
        if (my_col == 0) {
            if (cols > 1) {
                int east_neighbor = get_rank(cart_comm, 0, my_col + 1);
                MPI_Send(data, count, datatype, east_neighbor, 0, cart_comm);
                message_count++;
            }
        } else {
            int west_neighbor = get_rank(cart_comm, 0, my_col - 1);
            MPI_Recv(data, count, datatype, west_neighbor, 0, cart_comm, &status);

            if (my_col < cols - 1) {
                int east_neighbor = get_rank(cart_comm, 0, my_col + 1);
                MPI_Send(data, count, datatype, east_neighbor, 0, cart_comm);
                message_count++;
            }
        }
    }

    if (my_row == 0) {
        if (rows > 1) {
            int south_neighbor = get_rank(cart_comm, my_row + 1, my_col);
            MPI_Send(data, count, datatype, south_neighbor, 1, cart_comm);
            message_count++;
        }
    } else {
        int north_neighbor = get_rank(cart_comm, my_row - 1, my_col);
        MPI_Recv(data, count, datatype, north_neighbor, 1, cart_comm, &status);

        if (my_row < rows - 1) {
            int south_neighbor = get_rank(cart_comm, my_row + 1, my_col);
            MPI_Send(data, count, datatype, south_neighbor, 1, cart_comm);
            message_count++;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int dim_len = sqrt(world_size);
    if (dim_len * dim_len != world_size) {
        if (MPI_COMM_WORLD == 0) std::cerr << "Error: Square grid required." << std::endl;
        MPI_Finalize(); return 1;
    }

    int dims[2] = {dim_len, dim_len};
    int periods[2] = {0, 0};
    int reorder = 1;
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    int my_rank;
    int my_coords[2];
    MPI_Comm_rank(cart_comm, &my_rank);
    MPI_Cart_coords(cart_comm, my_rank, 2, my_coords);

    int val = 0;
    if (my_rank == 0) val = 999;

    double t1 = MPI_Wtime();
    bcast_row_col(&val, 1, MPI_INT, cart_comm, my_coords[0], my_coords[1], dims);
    double t2 = MPI_Wtime();

    double max_t2;
    MPI_Reduce(&t2, &max_t2, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    int local_error = (val != 999) ? 1 : 0;
    int total_errors = 0;
    MPI_Reduce(&local_error, &total_errors, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    int total_messages = 0;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    if (my_rank == 0) {
        printf("%d x %d grid:\n", dim_len, dim_len); 
        printf("Row-Column Broadcast (%dx%d) Completed.\n", dim_len, dim_len);
        printf("Time Taken: %f\n", max_t2 - t1);
        printf("Total messages sent: %d\n", total_messages);
        
        if (total_errors > 0) printf("FAILED! %d nodes had errors.\n", total_errors);
        else printf("SUCCESS!\n");
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
