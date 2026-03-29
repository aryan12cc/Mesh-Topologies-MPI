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

void send_to_children(int* data, int x, int y, int z, int dims[3], MPI_Comm comm, std::vector<MPI_Request>& reqs) {
    int dx = dims[0], dy = dims[1], dz = dims[2];
    
    if (x < dx - 1) {
        int child = get_rank_3d(comm, x + 1, y, z);
        MPI_Request req;
        MPI_Isend(data, 1, MPI_INT, child, 0, comm, &req);
        reqs.push_back(req);
        message_count++;  
    }
    
    if (y < dy - 1) {
        int child = get_rank_3d(comm, x, y + 1, z);
        MPI_Request req;
        MPI_Isend(data, 1, MPI_INT, child, 0, comm, &req);
        reqs.push_back(req);
        message_count++;  
    }

    if (z < dz - 1) {
        int child = get_rank_3d(comm, x, y, z + 1);
        MPI_Request req;
        MPI_Isend(data, 1, MPI_INT, child, 0, comm, &req);
        reqs.push_back(req);
        message_count++;  
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim = round(cbrt(size)); 
    if (dim * dim * dim != size) {
        if(rank==0) std::cerr << "Error: Process count must be a perfect cube." << std::endl;
        MPI_Finalize(); return 1;
    }

    int dims[3] = {dim, dim, dim};
    int periods[3] = {0, 0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);

    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    int x = coords[0]; 
    int y = coords[1]; 
    int z = coords[2];

    vector<int> parents;
    if (x > 0) parents.push_back(get_rank_3d(cart_comm, x - 1, y, z));
    if (y > 0) parents.push_back(get_rank_3d(cart_comm, x, y - 1, z));
    if (z > 0) parents.push_back(get_rank_3d(cart_comm, x, y, z - 1));

    int my_data = 0;
    
    vector<int> recv_buffers(parents.size(), 0);
    vector<MPI_Request> recv_reqs;
    vector<MPI_Request> send_reqs;

    double t1 = MPI_Wtime();

    if (rank == 0) {
        
        my_data = 999;
        send_to_children(&my_data, x, y, z, dims, cart_comm, send_reqs);
    } 
    else {
        
        for (size_t i = 0; i < parents.size(); i++) {
            MPI_Request req;
            MPI_Irecv(&recv_buffers[i], 1, MPI_INT, parents[i], 0, cart_comm, &req);
            recv_reqs.push_back(req);
        }

        int index_finished;
        MPI_Waitany(recv_reqs.size(), recv_reqs.data(), &index_finished, MPI_STATUS_IGNORE);

        my_data = recv_buffers[index_finished];

        send_to_children(&my_data, x, y, z, dims, cart_comm, send_reqs);
        
    }

    if (!send_reqs.empty()) {
        MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);
    }

    double t2 = MPI_Wtime();
    double max_time;
    MPI_Reduce(&t2, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    t2 = max_time;

    int expected_val = 999;
    int local_error = (my_data != expected_val) ? 1 : 0;
    int total_errors = 0;
    MPI_Reduce(&local_error, &total_errors, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    int total_messages = 0;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        printf("3D Wavefront Broadcast (%dx%dx%d) Completed.\n", dim, dim, dim);
        printf("Time Taken:     %f\n", t2 - t1);
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
