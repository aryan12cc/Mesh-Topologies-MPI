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

void send_to_down_and_right(int* data, int my_r, int my_c, int rows, int cols, 
                             MPI_Comm comm, std::vector<MPI_Request>& send_reqs) {
    if (my_r < rows - 1) {
        int south_neighbor = get_rank(comm, my_r + 1, my_c);
        MPI_Request req;
        MPI_Isend(data, 1, MPI_INT, south_neighbor, 0, comm, &req);
        send_reqs.push_back(req);
        message_count++;
    }

    if (my_c < cols - 1) {
        int east_neighbor = get_rank(comm, my_r, my_c + 1);
        MPI_Request req;
        MPI_Isend(data, 1, MPI_INT, east_neighbor, 0, comm, &req);
        send_reqs.push_back(req);
        message_count++;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dim = sqrt(size);

    if (dim * dim != size) {
        if (rank == 0) std::cerr << "Error: Number of processes must be a perfect square." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int dims[2] = {dim, dim};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int my_coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, my_coords);
    int r = my_coords[0];
    int c = my_coords[1];

    std::vector<int> parents;
    if (r > 0) parents.push_back(get_rank(cart_comm, r - 1, c));
    if (c > 0) parents.push_back(get_rank(cart_comm, r, c - 1));

    MPI_Barrier(cart_comm);
    
    double t_start = MPI_Wtime();

    int data = 0;
    std::vector<MPI_Request> send_reqs;

    if (rank == 0) {
        data = 999;
        send_to_down_and_right(&data, r, c, dim, dim, cart_comm, send_reqs);
    } else {
        std::vector<int> recv_buffers(parents.size());
        std::vector<MPI_Request> recv_reqs(parents.size());
        
        for (size_t i = 0; i < parents.size(); i++) {
            MPI_Irecv(&recv_buffers[i], 1, MPI_INT, parents[i], 0, cart_comm, &recv_reqs[i]);
        }

        int index;
        MPI_Waitany(recv_reqs.size(), recv_reqs.data(), &index, MPI_STATUS_IGNORE);
        
        data = recv_buffers[index];
        
        send_to_down_and_right(&data, r, c, dim, dim, cart_comm, send_reqs);
    }

    if (!send_reqs.empty()) {
        MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);
    }

    double t_end = MPI_Wtime();
    double max_end;
    MPI_Reduce(&t_end, &max_end, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    int local_error = (data != 999) ? 1 : 0;
    int total_errors = 0;
    MPI_Reduce(&local_error, &total_errors, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    int total_messages;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        printf("%d x %d grid:\n", dim, dim);
        printf("BFS Broadcast Complete: %f seconds\n", max_end - t_start);
        printf("Total messages sent: %d\n", total_messages);
        
        if (total_errors > 0) {
            printf("FAILED! %d nodes had errors.\n", total_errors);
        } else {
            printf("SUCCESS!\n");
        }
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
