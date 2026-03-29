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

void reduce_inclusion_exclusion_async_3d(int* local_val, int* global_result, MPI_Comm comm, int dims[3]) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int coords[3];
    MPI_Cart_coords(comm, rank, 3, coords);
    int x = coords[0];
    int y = coords[1];
    int z = coords[2];

    int dx = dims[0];
    int dy = dims[1];
    int dz = dims[2];

    int v_x = 0, v_y = 0, v_z = 0; 
    
    int v_xy = 0, v_xz = 0, v_yz = 0; 
    
    int v_xyz = 0; 

    vector<MPI_Request> recv_reqs;
    
    if (x < dx - 1) { 
        MPI_Request req;
        MPI_Irecv(&v_x, 1, MPI_INT, get_rank_3d(comm, x+1, y, z), 1, comm, &req);
        recv_reqs.push_back(req);
    }
    if (y < dy - 1) { 
        MPI_Request req;
        MPI_Irecv(&v_y, 1, MPI_INT, get_rank_3d(comm, x, y+1, z), 2, comm, &req);
        recv_reqs.push_back(req);
    }
    if (z < dz - 1) { 
        MPI_Request req;
        MPI_Irecv(&v_z, 1, MPI_INT, get_rank_3d(comm, x, y, z+1), 3, comm, &req);
        recv_reqs.push_back(req);
    }

    if (x < dx - 1 && y < dy - 1) { 
        MPI_Request req;
        MPI_Irecv(&v_xy, 1, MPI_INT, get_rank_3d(comm, x+1, y+1, z), 4, comm, &req);
        recv_reqs.push_back(req);
    }
    if (x < dx - 1 && z < dz - 1) { 
        MPI_Request req;
        MPI_Irecv(&v_xz, 1, MPI_INT, get_rank_3d(comm, x+1, y, z+1), 5, comm, &req);
        recv_reqs.push_back(req);
    }
    if (y < dy - 1 && z < dz - 1) { 
        MPI_Request req;
        MPI_Irecv(&v_yz, 1, MPI_INT, get_rank_3d(comm, x, y+1, z+1), 6, comm, &req);
        recv_reqs.push_back(req);
    }

    if (x < dx - 1 && y < dy - 1 && z < dz - 1) { 
        MPI_Request req;
        MPI_Irecv(&v_xyz, 1, MPI_INT, get_rank_3d(comm, x+1, y+1, z+1), 7, comm, &req);
        recv_reqs.push_back(req);
    }

    if (!recv_reqs.empty()) {
        MPI_Waitall(recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE);
    }

    int calculated_sum = *local_val 
                       + (v_x + v_y + v_z)          
                       - (v_xy + v_xz + v_yz)       
                       + (v_xyz);                   

    vector<MPI_Request> send_reqs;

    if (x > 0) {
        MPI_Request req;
        MPI_Isend(&calculated_sum, 1, MPI_INT, get_rank_3d(comm, x-1, y, z), 1, comm, &req);
        send_reqs.push_back(req);
        message_count++;  
    }
    if (y > 0) {
        MPI_Request req;
        MPI_Isend(&calculated_sum, 1, MPI_INT, get_rank_3d(comm, x, y-1, z), 2, comm, &req);
        send_reqs.push_back(req);
        message_count++;  
    }
    if (z > 0) {
        MPI_Request req;
        MPI_Isend(&calculated_sum, 1, MPI_INT, get_rank_3d(comm, x, y, z-1), 3, comm, &req);
        send_reqs.push_back(req);
        message_count++;  
    }

    if (x > 0 && y > 0) {
        MPI_Request req;
        MPI_Isend(&calculated_sum, 1, MPI_INT, get_rank_3d(comm, x-1, y-1, z), 4, comm, &req);
        send_reqs.push_back(req);
        message_count++;  
    }
    if (x > 0 && z > 0) {
        MPI_Request req;
        MPI_Isend(&calculated_sum, 1, MPI_INT, get_rank_3d(comm, x-1, y, z-1), 5, comm, &req);
        send_reqs.push_back(req);
        message_count++;  
    }
    if (y > 0 && z > 0) {
        MPI_Request req;
        MPI_Isend(&calculated_sum, 1, MPI_INT, get_rank_3d(comm, x, y-1, z-1), 6, comm, &req);
        send_reqs.push_back(req);
        message_count++;  
    }

    if (x > 0 && y > 0 && z > 0) {
        MPI_Request req;
        MPI_Isend(&calculated_sum, 1, MPI_INT, get_rank_3d(comm, x-1, y-1, z-1), 7, comm, &req);
        send_reqs.push_back(req);
        message_count++;  
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

    int dim = round(cbrt(size)); 
    if (dim * dim * dim != size) {
        if(rank==0) cerr << "Error: Process count must be a perfect cube." << endl;
        MPI_Finalize(); return 1;
    }

    int dims[3] = {dim, dim, dim};
    int periods[3] = {0, 0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);

    int my_val = 1;
    int final_sum = 0;

    double t1 = MPI_Wtime();
    
    reduce_inclusion_exclusion_async_3d(&my_val, &final_sum, cart_comm, dims);
    
    double t2 = MPI_Wtime();
    double tmax;
    MPI_Reduce(&t2, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    t2 = tmax;

    int total_messages = 0;
    MPI_Reduce(&message_count, &total_messages, 1, MPI_INT, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        printf("3D Async Inclusion-Exclusion Reduce (%dx%dx%d) Completed.\n", dim, dim, dim);
        printf("Expected Sum: %d\n", size);
        printf("Actual Sum:   %d\n", final_sum);
        printf("Time Taken:   %f\n", t2 - t1);
        printf("Total messages sent: %d\n", total_messages);
        
        if (final_sum != size) printf("FAILED!\n");
        else printf("SUCCESS!\n");
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
