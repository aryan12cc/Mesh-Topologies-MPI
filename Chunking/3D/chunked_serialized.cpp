#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstring>

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

int get_rank_3d(MPI_Comm comm, int x, int y, int z) {
    int coords[3] = {x, y, z};
    int rank;
    MPI_Cart_rank(comm, coords, &rank);
    return rank;
}

const int RECORD_SIZE = sizeof(int) + sizeof(double) + 32 + sizeof(int) * 8; 

void bcast_xyz_chunked_serialized(char* serialized_data, int total_records, int chunk_records,
                                   MPI_Comm comm, int dims[3]) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int coords[3];
    MPI_Cart_coords(comm, rank, 3, coords);
    int my_x = coords[0];
    int my_y = coords[1];
    int my_z = coords[2];

    int dim_x = dims[0];
    int dim_y = dims[1];
    int dim_z = dims[2];

    MPI_Status status;
    
    int num_chunks = (total_records + chunk_records - 1) / chunk_records;
    
    for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        int records_in_chunk = (chunk_id == num_chunks - 1) ? 
                               (total_records - chunk_id * chunk_records) : chunk_records;
        int chunk_byte_size = records_in_chunk * RECORD_SIZE;
        int chunk_offset = chunk_id * chunk_records * RECORD_SIZE;
        char* chunk_ptr = serialized_data + chunk_offset;
        
        if (my_y == 0 && my_z == 0) {
            if (my_x > 0) {
                int source = get_rank_3d(comm, my_x - 1, my_y, my_z);
                MPI_Recv(chunk_ptr, chunk_byte_size, MPI_CHAR, source, chunk_id, comm, &status);
            }
            
            if (my_x < dim_x - 1) {
                int dest = get_rank_3d(comm, my_x + 1, my_y, my_z);
                MPI_Send(chunk_ptr, chunk_byte_size, MPI_CHAR, dest, chunk_id, comm);
            }
        }

        if (my_z == 0) {
            if (my_y > 0) {
                int source = get_rank_3d(comm, my_x, my_y - 1, my_z);
                MPI_Recv(chunk_ptr, chunk_byte_size, MPI_CHAR, source, chunk_id + 10000, comm, &status);
            }
            
            if (my_y < dim_y - 1) {
                int dest = get_rank_3d(comm, my_x, my_y + 1, my_z);
                MPI_Send(chunk_ptr, chunk_byte_size, MPI_CHAR, dest, chunk_id + 10000, comm);
            }
        }

        if (my_z > 0) {
            int source = get_rank_3d(comm, my_x, my_y, my_z - 1);
            MPI_Recv(chunk_ptr, chunk_byte_size, MPI_CHAR, source, chunk_id + 20000, comm, &status);
        }

        if (my_z < dim_z - 1) {
            int dest = get_rank_3d(comm, my_x, my_y, my_z + 1);
            MPI_Send(chunk_ptr, chunk_byte_size, MPI_CHAR, dest, chunk_id + 20000, comm);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int dim_len = round(cbrt(world_size));
    if (dim_len * dim_len * dim_len != world_size) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) std::cerr << "Error: Need cubic number of procs." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int dims[3] = {dim_len, dim_len, dim_len};
    int periods[3] = {0, 0, 0};
    int reorder = 1;
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cart_comm);

    int my_rank;
    int my_coords[3];
    MPI_Comm_rank(cart_comm, &my_rank);
    MPI_Cart_coords(cart_comm, my_rank, 3, my_coords);

    Record* records = new Record[NUM_RECORDS];
    
    if (my_coords[0] == 0 && my_coords[1] == 0 && my_coords[2] == 0) {
        for (int i = 0; i < NUM_RECORDS; i++) {
            records[i].id = i;
            records[i].value = i * 3.14159;
            snprintf(records[i].name, 32, "Record_%d", i);
            for (int j = 0; j < 8; j++) {
                records[i].data[j] = i * 10 + j;
            }
        }
    }
    
    int total_bytes = NUM_RECORDS * RECORD_SIZE;
    char* serialized = new char[total_bytes];
    memset(serialized, 0, total_bytes);
    
    double serialize_start = MPI_Wtime();
    if (my_coords[0] == 0 && my_coords[1] == 0 && my_coords[2] == 0) {
        int offset = 0;
        for (int i = 0; i < NUM_RECORDS; i++) {
            serialize_record(records[i], serialized, offset);
        }
    }
    double serialize_end = MPI_Wtime();
    
    MPI_Barrier(cart_comm);
    double bcast_start = MPI_Wtime();
    
    bcast_xyz_chunked_serialized(serialized, NUM_RECORDS, CHUNK_RECORDS, cart_comm, dims);
    
    double bcast_end = MPI_Wtime();
    
    double deserialize_start = MPI_Wtime();
    if (!(my_coords[0] == 0 && my_coords[1] == 0 && my_coords[2] == 0)) {
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
        printf("ERROR: Rank %d (Coords %d,%d,%d) failed verification.\n", 
               my_rank, my_coords[0], my_coords[1], my_coords[2]);
    }
    
    if (my_rank == 0) {
        printf("Chunked 3D Serialized Broadcast:\n");
        printf("  Records: %d, Total bytes: %d, Chunks: %d\n", 
               NUM_RECORDS, total_bytes, (NUM_RECORDS + CHUNK_RECORDS - 1) / CHUNK_RECORDS);
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
