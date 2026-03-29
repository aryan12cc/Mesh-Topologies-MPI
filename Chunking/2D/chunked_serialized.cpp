#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <sstream>

#define NUM_RECORDS (1 << 22)  
#define CHUNK_RECORDS (1 << 18) 

struct Record {
    int id;
    double value;
    char name[32];
    int data[8];
};

void serialize_record(const Record& rec, char* buffer, int& offset) {
    memcpy(buffer + offset, &rec.id, sizeof(int));
    offset += sizeof(int);
    memcpy(buffer + offset, &rec.value, sizeof(double));
    offset += sizeof(double);
    memcpy(buffer + offset, rec.name, 32);
    offset += 32;
    memcpy(buffer + offset, rec.data, sizeof(int) * 8);
    offset += sizeof(int) * 8;
}

void deserialize_record(Record& rec, const char* buffer, int& offset) {
    memcpy(&rec.id, buffer + offset, sizeof(int));
    offset += sizeof(int);
    memcpy(&rec.value, buffer + offset, sizeof(double));
    offset += sizeof(double);
    memcpy(rec.name, buffer + offset, 32);
    offset += 32;
    memcpy(rec.data, buffer + offset, sizeof(int) * 8);
    offset += sizeof(int) * 8;
}

int get_rank_from_coords(MPI_Comm comm, int row, int col) {
    int coords[2] = {row, col};
    int rank;
    MPI_Cart_rank(comm, coords, &rank);
    return rank;
}

const int RECORD_SIZE = sizeof(int) + sizeof(double) + 32 + sizeof(int) * 8; 

void bcast_chunked_serialized(char* serialized_data, int total_records, int chunk_records,
                               MPI_Comm cart_comm, int my_row, int my_col, int dims[2]) {
    int rows = dims[0];
    int cols = dims[1];
    
    int num_chunks = (total_records + chunk_records - 1) / chunk_records;
    
    MPI_Request* send_reqs = new MPI_Request[num_chunks * 2]; 
    MPI_Request* recv_reqs = new MPI_Request[num_chunks * 2]; 
    int send_count = 0;
    int recv_count = 0;
    
    for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        int records_in_chunk = (chunk_id == num_chunks - 1) ? 
                               (total_records - chunk_id * chunk_records) : chunk_records;
        int chunk_byte_size = records_in_chunk * RECORD_SIZE;
        int chunk_offset = chunk_id * chunk_records * RECORD_SIZE;
        char* chunk_ptr = serialized_data + chunk_offset;
        
        if (my_row == 0) {
            if (my_col == 0) {
                
                if (cols > 1) {
                    int east_neighbor = get_rank_from_coords(cart_comm, 0, my_col + 1);
                    MPI_Isend(chunk_ptr, chunk_byte_size, MPI_CHAR, east_neighbor, chunk_id, cart_comm, &send_reqs[send_count++]);
                }
            } else {
                
                int west_neighbor = get_rank_from_coords(cart_comm, 0, my_col - 1);
                MPI_Irecv(chunk_ptr, chunk_byte_size, MPI_CHAR, west_neighbor, chunk_id, cart_comm, &recv_reqs[recv_count++]);
                
                MPI_Wait(&recv_reqs[recv_count - 1], MPI_STATUS_IGNORE);
                
                if (my_col < cols - 1) {
                    int east_neighbor = get_rank_from_coords(cart_comm, 0, my_col + 1);
                    MPI_Isend(chunk_ptr, chunk_byte_size, MPI_CHAR, east_neighbor, chunk_id, cart_comm, &send_reqs[send_count++]);
                }
            }
            
            if (rows > 1) {
                int south_neighbor = get_rank_from_coords(cart_comm, my_row + 1, my_col);
                MPI_Isend(chunk_ptr, chunk_byte_size, MPI_CHAR, south_neighbor, chunk_id + 10000, cart_comm, &send_reqs[send_count++]);
            }
        } else {
            
            int north_neighbor = get_rank_from_coords(cart_comm, my_row - 1, my_col);
            MPI_Irecv(chunk_ptr, chunk_byte_size, MPI_CHAR, north_neighbor, chunk_id + 10000, cart_comm, &recv_reqs[recv_count++]);
            
            MPI_Wait(&recv_reqs[recv_count - 1], MPI_STATUS_IGNORE);
            
            if (my_row < rows - 1) {
                int south_neighbor = get_rank_from_coords(cart_comm, my_row + 1, my_col);
                MPI_Isend(chunk_ptr, chunk_byte_size, MPI_CHAR, south_neighbor, chunk_id + 10000, cart_comm, &send_reqs[send_count++]);
            }
        }
    }
    
    if (send_count > 0) {
        MPI_Waitall(send_count, send_reqs, MPI_STATUSES_IGNORE);
    }
    
    delete[] send_reqs;
    delete[] recv_reqs;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int dim_len = sqrt(world_size);
    if (dim_len * dim_len != world_size) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) std::cerr << "Error: Need square number of procs." << std::endl;
        MPI_Finalize();
        return 1;
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

    Record* records = new Record[NUM_RECORDS];
    
    if (my_coords[0] == 0 && my_coords[1] == 0) {
        for (int i = 0; i < NUM_RECORDS; i++) {
            records[i].id = i;
            records[i].value = i * 3.14159;
            snprintf(records[i].name, 32, "Record_%d", i);
            for (int j = 0; j < 8; j++) {
                records[i].data[j] = i * 10 + j;
            }
        }
    }
    
    long long total_bytes = (long long) NUM_RECORDS * (long long) RECORD_SIZE;
    char* serialized = new char[total_bytes];
    memset(serialized, 0, total_bytes);
    
    double serialize_start = MPI_Wtime();
    if (my_coords[0] == 0 && my_coords[1] == 0) {
        int offset = 0;
        for (int i = 0; i < NUM_RECORDS; i++) {
            serialize_record(records[i], serialized, offset);
        }
    }
    double serialize_end = MPI_Wtime();
    
    MPI_Barrier(cart_comm);
    double bcast_start = MPI_Wtime();
    
    bcast_chunked_serialized(serialized, NUM_RECORDS, CHUNK_RECORDS, 
                             cart_comm, my_coords[0], my_coords[1], dims);
    
    double bcast_end = MPI_Wtime();
    
    double deserialize_start = MPI_Wtime();
    if (!(my_coords[0] == 0 && my_coords[1] == 0)) {
        int offset = 0;
        for (int i = 0; i < NUM_RECORDS; i++) {
            deserialize_record(records[i], serialized, offset);
        }
    }
    double deserialize_end = MPI_Wtime();
    
    double max_bcast_time;
    MPI_Reduce(&bcast_end, &max_bcast_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    
    bool error = false;
    for (int i = 0; i < NUM_RECORDS && !error; i++) {
        if (records[i].id != i || 
            fabs(records[i].value - i * 3.14159) > 0.0001) {
            error = true;
            printf("ERROR at rank %d: Record %d has id=%d, value=%f\n", 
                   my_rank, i, records[i].id, records[i].value);
        }
    }
    
    if (error) {
        printf("ERROR: Rank %d (Coords %d,%d) failed verification.\n", 
               my_rank, my_coords[0], my_coords[1]);
    }
    
    if (my_rank == 0) {
        int num_chunks = (NUM_RECORDS + CHUNK_RECORDS - 1) / CHUNK_RECORDS;
        printf("Chunked Serialized Broadcast:\n");
        printf("  Records: %d, Chunk size: %d records, Chunks: %d\n", 
               NUM_RECORDS, CHUNK_RECORDS, num_chunks);
        printf("  Serialize time: %f seconds\n", serialize_end - serialize_start);
        printf("  Broadcast time: %f seconds\n", max_bcast_time - bcast_start);
        printf("  Deserialize time: %f seconds\n", deserialize_end - deserialize_start);
        printf("  Total time: %f seconds\n", 
               (serialize_end - serialize_start) + (max_bcast_time - bcast_start) + 
               (deserialize_end - deserialize_start));
    }

    delete[] records;
    delete[] serialized;
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
