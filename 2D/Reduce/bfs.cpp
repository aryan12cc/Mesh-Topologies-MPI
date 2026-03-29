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

void reduce_inclusion_exclusion_async(int* local_val, int* global_result, MPI_Comm comm, int grid_dim) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int coords[2];
    MPI_Cart_coords(comm, rank, 2, coords);
    int my_r = coords[0];
    int my_c = coords[1];

    int v_south = 0, v_east = 0, v_diag = 0;

    std::vector<MPI_Request> recv_reqs;
    
    if (my_r < grid_dim - 1) {
        int south_rank = get_rank(comm, my_r + 1, my_c);
        MPI_Request req;
        MPI_Irecv(&v_south, 1, MPI_INT, south_rank, 1, comm, &req);
        recv_reqs.push_back(req);
    }

    if (my_c < grid_dim - 1) {
        int east_rank = get_rank(comm, my_r, my_c + 1);
        MPI_Request req;
        MPI_Irecv(&v_east, 1, MPI_INT, east_rank, 2, comm, &req);
        recv_reqs.push_back(req);
    }

    if (my_r < grid_dim - 1 && my_c < grid_dim - 1) {
        int diag_rank = get_rank(comm, my_r + 1, my_c + 1);
        MPI_Request req;
        MPI_Irecv(&v_diag, 1, MPI_INT, diag_rank, 3, comm, &req);
        recv_reqs.push_back(req);
    }

    if (!recv_reqs.empty()) {
        MPI_Waitall(recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE);
    }

    int calculated_sum = *local_val + v_south + v_east - v_diag;

    std::vector<MPI_Request> send_reqs;

    if (my_r > 0) {
        int north_rank = get_rank(comm, my_r - 1, my_c);
        MPI_Request req;
        MPI_Isend(&calculated_sum, 1, MPI_INT, north_rank, 1, comm, &req);
        message_count++;
        send_reqs.push_back(req);
    }

    if (my_c > 0) {
        int west_rank = get_rank(comm, my_r, my_c - 1);
        MPI_Request req;
        MPI_Isend(&calculated_sum, 1, MPI_INT, west_rank, 2, comm, &req);
        message_count++;
        send_reqs.push_back(req);
    }

    if (my_r > 0 && my_c > 0) {
        int nw_rank = get_rank(comm, my_r - 1, my_c - 1);
        MPI_Request req;
        MPI_Isend(&calculated_sum, 1, MPI_INT, nw_rank, 3, comm, &req);
        message_count++;
        send_reqs.push_back(req);
    }

    if (!send_reqs.empty()) {
        MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);
    }

    if (rank == 0) {
        *global_result = calculated_sum;
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
    int dims[2] = {dim, dim};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int my_val = 1;
    int final_sum = 0;

    double t1 = MPI_Wtime();
    
    reduce_inclusion_exclusion_async(&my_val, &final_sum, cart_comm, dim);
    
    double t2 = MPI_Wtime();

    double max_time;
    MPI_Reduce(&t2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    t2 = max_time;

    int total_messages = 0;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        printf("Async Inclusion-Exclusion Reduce Completed.\n");
        printf("Time Taken:   %f\n", t2 - t1);
        printf("Total messages sent: %d\n", total_messages);
        
        if (final_sum != size) printf("FAILED!\n");
    }

    MPI_Finalize();
    return 0;
}
