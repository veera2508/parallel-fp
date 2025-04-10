#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <mpi.h>
#include <omp.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define HASH_SIZE 1073741824  // Must be power of 2
#define MAX_STATION_NAME 100

// Structure to hold temperature statistics
typedef struct {
    char name[MAX_STATION_NAME];
    double min;
    double max;
    double sum;
    long count;
    int initialized;
} StationStats;

// Hash table for station statistics
typedef struct {
    StationStats* table;
    int size;
    int count;
} HashTable;

// Initialize hash table
void init_hash_table(HashTable* ht, int size) {
    ht->table = (StationStats*)calloc(size, sizeof(StationStats));
    ht->size = size;
    ht->count = 0;
    
    for (int i = 0; i < size; i++) {
        ht->table[i].initialized = 0;
    }
}

// Simple hash function for station names
unsigned int hash_station(const char* station, int table_size) {
    unsigned int hash = 0;
    while (*station) {
        hash = hash * 31 + *station++;
    }
    return hash & (table_size - 1);  // Fast modulo for power of 2
}

// Find or create a station entry in the hash table
StationStats* find_or_create_station(HashTable* ht, const char* station) {
    unsigned int index = hash_station(station, ht->size);
    unsigned int original_index = index;
    
    // Linear probing to handle collisions
    while (ht->table[index].initialized && strcmp(ht->table[index].name, station) != 0) {
        index = (index + 1) & (ht->size - 1);
        if (index == original_index) {
            // Hash table is full, this shouldn't happen with proper sizing
            fprintf(stderr, "Hash table is full!\n");
            exit(1);
        }
    }
    
    // Initialize if this is a new entry
    if (!ht->table[index].initialized) {
        strncpy(ht->table[index].name, station, MAX_STATION_NAME - 1);
        ht->table[index].name[MAX_STATION_NAME - 1] = '\0';
        ht->table[index].min = DBL_MAX;
        ht->table[index].max = -DBL_MAX;
        ht->table[index].sum = 0.0;
        ht->table[index].count = 0;
        ht->table[index].initialized = 1;
        ht->count++;
    }
    
    return &ht->table[index];
}

// Fast string to double conversion
double fast_atof(const char* str) {
    double val = 0.0;
    double sign = 1.0;
    double power = 1.0;
    int decimal = 0;
    
    // Handle negative numbers
    if (*str == '-') {
        sign = -1.0;
        str++;
    }
    
    // Parse digits
    while (*str) {
        if (*str == '.') {
            decimal = 1;
        } else if (isdigit(*str)) {
            if (decimal) {
                power *= 0.1;
                val += (*str - '0') * power;
            } else {
                val = val * 10.0 + (*str - '0');
            }
        }
        str++;
    }
    
    return sign * val;
}

// Process a chunk of the file
void process_chunk(const char* file_path, long start_pos, long end_pos, HashTable* ht) {
    int fd = open(file_path, O_RDONLY);
    if (fd == -1) {
        perror("Error opening file");
        exit(1);
    }
    
    // Memory map the file
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Error getting file size");
        close(fd);
        exit(1);
    }
    
    char* file_data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_data == MAP_FAILED) {
        perror("Error memory mapping file");
        close(fd);
        exit(1);
    }
    
    // Adjust start position to beginning of a line
    if (start_pos > 0) {
        while (start_pos < sb.st_size && file_data[start_pos - 1] != '\n') {
            start_pos++;
        }
    }
    
    // Process each line in the chunk
    char station[MAX_STATION_NAME];
    char temp_str[20];
    int station_idx, temp_idx;
    long pos;
    
    #pragma omp parallel private(station, temp_str, station_idx, temp_idx, pos)
    {
        // Thread-local hash table
        HashTable local_ht;
        init_hash_table(&local_ht, HASH_SIZE);
        
        // Fixed loop structure for OpenMP
        #pragma omp for schedule(dynamic, 1000000)
        for (long i = 0; i < (end_pos - start_pos); i++) {
            pos = start_pos + i;
            
            // Skip if we've reached the end of the file
            if (pos >= sb.st_size) continue;
            
            // Find the start of the next line
            while (pos < end_pos && pos < sb.st_size && 
                  (file_data[pos] == '\n' || (pos > 0 && file_data[pos-1] != '\n'))) {
                pos++;
            }
            
            if (pos >= end_pos || pos >= sb.st_size) continue;
            
            // Parse station name
            station_idx = 0;
            while (pos < end_pos && pos < sb.st_size && file_data[pos] != ';') {
                if (station_idx < MAX_STATION_NAME - 1) {
                    station[station_idx++] = file_data[pos];
                }
                pos++;
            }
            station[station_idx] = '\0';
            
            // Skip semicolon
            if (pos < end_pos && pos < sb.st_size && file_data[pos] == ';') {
                pos++;
            } else {
                continue; // Malformed line
            }
            
            // Parse temperature
            temp_idx = 0;
            while (pos < end_pos && pos < sb.st_size && file_data[pos] != '\n') {
                if (temp_idx < 19) {
                    temp_str[temp_idx++] = file_data[pos];
                }
                pos++;
            }
            temp_str[temp_idx] = '\0';
            
            // Skip newline
            if (pos < end_pos && pos < sb.st_size && file_data[pos] == '\n') {
                pos++;
            }
            
            // Convert temperature and update statistics
            double temp = fast_atof(temp_str);
            StationStats* stats = find_or_create_station(&local_ht, station);
            
            if (temp < stats->min) stats->min = temp;
            if (temp > stats->max) stats->max = temp;
            stats->sum += temp;
            stats->count++;
        }
        
        // Merge thread-local results into the shared hash table
        #pragma omp critical
        {
            for (int i = 0; i < local_ht.size; i++) {
                if (local_ht.table[i].initialized) {
                    StationStats* global_stats = find_or_create_station(ht, local_ht.table[i].name);
                    
                    if (local_ht.table[i].min < global_stats->min) 
                        global_stats->min = local_ht.table[i].min;
                    
                    if (local_ht.table[i].max > global_stats->max) 
                        global_stats->max = local_ht.table[i].max;
                    
                    global_stats->sum += local_ht.table[i].sum;
                    global_stats->count += local_ht.table[i].count;
                }
            }
        }
        
        // Free thread-local hash table
        free(local_ht.table);
    }
    
    // Clean up
    munmap(file_data, sb.st_size);
    close(fd);
}


// Merge hash tables from different MPI processes
void merge_hash_tables(HashTable* local_ht, int rank, int size) {
    // Structure for sending/receiving station data
    typedef struct {
        char name[MAX_STATION_NAME];
        double min;
        double max;
        double sum;
        long count;
    } StationData;
    
    if (rank == 0) {
        // Master process receives and merges data from all other processes
        for (int src = 1; src < size; src++) {
            int num_stations;
            MPI_Status status;
            
            // Receive number of stations from this process
            MPI_Recv(&num_stations, 1, MPI_INT, src, 0, MPI_COMM_WORLD, &status);
            
            // Receive each station's data
            for (int i = 0; i < num_stations; i++) {
                StationData data;
                MPI_Recv(&data, sizeof(StationData), MPI_BYTE, src, 1, MPI_COMM_WORLD, &status);
                
                // Merge with local data
                StationStats* stats = find_or_create_station(local_ht, data.name);
                if (data.min < stats->min) stats->min = data.min;
                if (data.max > stats->max) stats->max = data.max;
                stats->sum += data.sum;
                stats->count += data.count;
            }
        }
    } else {
        // Worker processes send their data to the master
        int num_stations = 0;
        
        // Count initialized stations
        for (int i = 0; i < local_ht->size; i++) {
            if (local_ht->table[i].initialized) {
                num_stations++;
            }
        }
        
        // Send count to master
        MPI_Send(&num_stations, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        
        // Send each station's data
        for (int i = 0; i < local_ht->size; i++) {
            if (local_ht->table[i].initialized) {
                StationData data;
                strncpy(data.name, local_ht->table[i].name, MAX_STATION_NAME);
                data.min = local_ht->table[i].min;
                data.max = local_ht->table[i].max;
                data.sum = local_ht->table[i].sum;
                data.count = local_ht->table[i].count;
                
                MPI_Send(&data, sizeof(StationData), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
            }
        }
    }
}

// Compare function for qsort
int compare_stations(const void* a, const void* b) {
    return strcmp(((StationStats*)a)->name, ((StationStats*)b)->name);
}

// Print results in alphabetical order
void print_results(HashTable* ht) {
    // Copy initialized entries to an array for sorting
    StationStats* stations = (StationStats*)malloc(ht->count * sizeof(StationStats));
    int idx = 0;
    
    for (int i = 0; i < ht->size; i++) {
        if (ht->table[i].initialized) {
            stations[idx++] = ht->table[i];
        }
    }
    
    // Sort stations alphabetically
    qsort(stations, ht->count, sizeof(StationStats), compare_stations);
    
    // Print results
    printf("{");
    for (int i = 0; i < ht->count; i++) {
        double mean = stations[i].sum / stations[i].count;
        printf("%s=%.1f/%.1f/%.1f", 
               stations[i].name, 
               stations[i].min, 
               mean, 
               stations[i].max);
        
        if (i < ht->count - 1) {
            printf(", ");
        }
    }
    printf("}\n");
    
    free(stations);
}

int main(int argc, char* argv[]) {
    int rank, size;
    double start_time, end_time;
    
    // Initialize MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        start_time = MPI_Wtime();
    }
    
    // Check command line arguments
    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    const char* file_path = argv[1];
    
    // Get file size
    struct stat sb;
    if (stat(file_path, &sb) == -1) {
        if (rank == 0) {
            perror("Error getting file size");
        }
        MPI_Finalize();
        return 1;
    }
    long file_size = sb.st_size;
    
    // Calculate chunk size for each MPI process
    long chunk_size = file_size / size;
    long start_pos = rank * chunk_size;
    long end_pos = (rank == size - 1) ? file_size : (rank + 1) * chunk_size;
    
    // Initialize hash table
    HashTable ht;
    init_hash_table(&ht, HASH_SIZE);
    
    // Process assigned chunk
    process_chunk(file_path, start_pos, end_pos, &ht);
    
    // Merge results from all processes
    merge_hash_tables(&ht, rank, size);
    
    // Print results (only from rank 0)
    if (rank == 0) {
        print_results(&ht);
        end_time = MPI_Wtime();
        printf("Execution time: %.2f seconds\n", end_time - start_time);
    }
    
    // Clean up
    free(ht.table);
    MPI_Finalize();
    
    return 0;
}
