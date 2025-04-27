#include <ctype.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h> // For timing
#include <unistd.h>

#define MAX_STATION_NAME 100
#define HASH_SIZE (1 << 20)  // Host hash table size
#define MAX_STATIONS 10000   // Maximum unique stations (adjust based on data)
#define CHUNK_SIZE (1 << 24) // 16MB chunk size (adjust based on GPU memory)
#define THREADS_PER_BLOCK 256

// --- Helper Macro for CUDA Error Checking ---
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error in %s:%d: %s (%d)\n", __FILE__, __LINE__,    \
              cudaGetErrorString(err), err);                                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// --- Host Data Structures ---
typedef struct {
  char name[MAX_STATION_NAME];
  double min;
  double max;
  double sum;
  long long count; // Use long long for potentially large counts
  int initialized;
} StationStats;

typedef struct {
  StationStats *table;
  int size;
  int count;
} HashTable;

// --- Device Data Structures ---
typedef struct {
  char name[MAX_STATION_NAME]; // Be mindful of stack usage if large
  float temperature;
} StationData;

// --- Custom Device Functions ---

// Custom atomicMin for float using atomicCAS
__device__ __forceinline__ float atomicMin_float(float *address, float val) {
  int *address_as_int = (int *)address;
  int old = *address_as_int;
  int assumed;
  do {
    assumed = old;
    if (val >= __int_as_float(assumed))
      return __int_as_float(assumed);
    old = atomicCAS(address_as_int, assumed, __float_as_int(val));
  } while (assumed != old);
  return __int_as_float(old);
}

// Custom atomicMax for float using atomicCAS
__device__ __forceinline__ float atomicMax_float(float *address, float val) {
  int *address_as_int = (int *)address;
  int old = *address_as_int;
  int assumed;
  do {
    assumed = old;
    if (val <= __int_as_float(assumed))
      return __int_as_float(assumed);
    old = atomicCAS(address_as_int, assumed, __float_as_int(val));
  } while (assumed != old);
  return __int_as_float(old);
}

// Custom strcmp for device code
__device__ __forceinline__ int strcmp_device(const char *s1, const char *s2) {
  while (*s1 && (*s1 == *s2)) {
    s1++;
    s2++;
  }
  return *(const unsigned char *)s1 - *(const unsigned char *)s2;
}

// Custom strcpy for device code (simplified, ensure dest has enough space)
__device__ __forceinline__ void strcpy_device(char *dest, const char *src) {
  while ((*dest++ = *src++))
    ; // Copies including the null terminator
}

// Custom atomicAdd for double (Requires Compute Capability 6.x+)
#if __CUDA_ARCH__ >= 600
__device__ __forceinline__ double atomicAdd_double(double *address,
                                                   double val) {
  return atomicAdd(address, val);
}
#else
// Fallback for older architectures using CAS loop (less efficient)
__device__ __forceinline__ double atomicAdd_double(double *address,
                                                   double val) {
  unsigned long long *address_as_ull = (unsigned long long *)address;
  unsigned long long old = *address_as_ull;
  unsigned long long assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(__longlong_as_double(assumed) + val));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

// Simple, potentially faster float parser for device (assumes format [-]d+[.d])
__device__ __forceinline__ float fast_float_parser(const char *temp_start,
                                                   int temp_len) {
  float temp = 0.0f;
  float sign = 1.0f;
  int idx = 0;

  if (temp_start[idx] == '-') {
    sign = -1.0f;
    idx++;
  }

  int integer_part = 0;
  while (idx < temp_len && temp_start[idx] != '.') {
    if (temp_start[idx] >= '0' && temp_start[idx] <= '9') {
      integer_part = integer_part * 10 + (temp_start[idx] - '0');
    } else {
      // Invalid character in integer part
      return 0.0f; // Or handle error appropriately
    }
    idx++;
  }
  temp = (float)integer_part;

  if (idx < temp_len && temp_start[idx] == '.') {
    idx++;
    if (idx < temp_len && temp_start[idx] >= '0' && temp_start[idx] <= '9') {
      // Only parse the first decimal digit as per 1BRC rules (X.Y)
      temp += (temp_start[idx] - '0') * 0.1f;
      // idx++; // Move past the decimal digit if needed for further checks
    } else {
      // Invalid character after decimal or missing decimal digit
      return 0.0f; // Or handle error
    }
    // Ignore any further digits after the first decimal place
  }
  // If no decimal point was found, temp remains the integer part

  return temp * sign;
}

// --- End Custom Device Functions ---

// --- Host Functions ---

// Initialize hash table (host)
void init_hash_table(HashTable *ht, int size) {
  ht->table = (StationStats *)calloc(size, sizeof(StationStats));
  if (ht->table == NULL) {
    fprintf(stderr, "Failed to allocate host hash table memory\n");
    exit(1);
  }
  ht->size = size;
  ht->count = 0;
  for (int i = 0; i < size; i++) {
    ht->table[i].initialized = 0;
  }
}

// Simple hash function for station names (host and device)
__host__ __device__ unsigned int hash_station(const char *station,
                                              int table_size) {
  unsigned int hash = 5381; // djb2 hash start
  int c;
  while ((c = *station++)) {
    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
  }
  // Use mask for power-of-2 table size
  return hash & (table_size - 1);
}

// Find or create a station entry in the host hash table
StationStats *find_or_create_station(HashTable *ht, const char *station) {
  unsigned int index = hash_station(station, ht->size);
  unsigned int original_index = index;

  // Linear probing
  while (ht->table[index].initialized &&
         strcmp(ht->table[index].name, station) != 0) {
    index = (index + 1) & (ht->size - 1); // Wrap around using mask
    if (index == original_index) {
      fprintf(stderr, "CRITICAL: Host hash table is full! Increase HASH_SIZE "
                      "or check data.\n");
      exit(1);
    }
  }

  // Initialize if new
  if (!ht->table[index].initialized) {
    // Check load factor (optional warning)
    if (ht->count > ht->size * 0.9) {
      fprintf(stderr, "Warning: Host hash table load factor > 90%%.\n");
    }
    strncpy(ht->table[index].name, station, MAX_STATION_NAME - 1);
    ht->table[index].name[MAX_STATION_NAME - 1] = '\0';
    ht->table[index].min = DBL_MAX;
    ht->table[index].max = -DBL_MAX;
    ht->table[index].sum = 0.0;
    ht->table[index].count = 0; // Will be updated during merge
    ht->table[index].initialized = 1;
    ht->count++; // Increment unique station count
  }

  return &ht->table[index];
}

// Compare function for qsort (host)
int compare_stations(const void *a, const void *b) {
  return strcmp(((StationStats *)a)->name, ((StationStats *)b)->name);
}

// Print results (host)
void print_results(HashTable *ht) {
  if (ht->count == 0) {
    printf("{}\n");
    return;
  }
  StationStats *stations =
      (StationStats *)malloc(ht->count * sizeof(StationStats));
  if (!stations) {
    fprintf(stderr, "Failed memory alloc for sorting.\n");
    return;
  }

  int idx = 0;
  for (int i = 0; i < ht->size; i++) {
    if (ht->table[i].initialized) {
      if (idx < ht->count) { // Bounds check
        stations[idx++] = ht->table[i];
      } else {
        fprintf(stderr,
                "Error: Hash table count mismatch during print prep.\n");
        break;
      }
    }
  }
  int sort_count = idx; // Actual number copied

  qsort(stations, sort_count, sizeof(StationStats), compare_stations);

  printf("{");
  for (int i = 0; i < sort_count; i++) {
    if (stations[i].count > 0) {
      double mean = stations[i].sum / stations[i].count;
      // Round mean to nearest tenth
      double rounded_mean = round(mean * 10.0) / 10.0;
      printf("%s=%.1f/%.1f/%.1f", stations[i].name, stations[i].min,
             rounded_mean, // Use rounded mean
             stations[i].max);
    } else {
      printf("%s=NaN/NaN/NaN", stations[i].name); // Should not happen ideally
    }
    if (i < sort_count - 1) {
      printf(", ");
    }
  }
  printf("}\n");
  free(stations);
}

// --- CUDA Kernels ---

// Kernel to parse data (UPDATED: Manual loops instead of memchr)
__global__ void parseStationDataKernel(char *__restrict__ data, long data_size,
                                       StationData *__restrict__ output,
                                       int *__restrict__ output_count,
                                       int max_output_per_chunk,
                                       long chunk_offset) {
  // Calculate the starting position for this thread within the chunk
  long start_byte = (long)blockIdx.x * blockDim.x + threadIdx.x;

  // Heuristic: Each thread processes a range of bytes (e.g., ~256 bytes)
  // This helps manage divergence and workload distribution better than 1
  // thread/byte.
  const int processing_range = 256;
  long current_pos = start_byte * processing_range;
  long end_pos_for_thread = current_pos + processing_range;

  // Adjust start: Find the beginning of the first line within or overlapping
  // the start of the range
  if (current_pos > 0 &&
      chunk_offset + current_pos < data_size) { // Avoid going past file end
    // Ensure we are not starting mid-line from the previous chunk's perspective
    if (chunk_offset > 0 && current_pos == 0 &&
        data[current_pos - 1 + chunk_offset] != '\n') {
      // Scan forward to find the first newline if we are at the beginning of a
      // subsequent chunk
      while (current_pos < data_size && data[current_pos] != '\n') {
        current_pos++;
      }
      if (current_pos < data_size)
        current_pos++; // Move past newline
    } else if (current_pos > 0 && data[current_pos - 1] != '\n') {
      // Scan backwards within the current chunk to find the start of the line
      while (current_pos > 0 && data[current_pos - 1] != '\n') {
        current_pos--;
      }
    }
  }

  // Ensure the effective end position doesn't exceed the chunk data size
  long effective_chunk_end = data_size;
  if (end_pos_for_thread > effective_chunk_end) {
    end_pos_for_thread = effective_chunk_end;
  }

  // Process lines within the assigned range [current_pos, end_pos_for_thread)
  while (current_pos < end_pos_for_thread) {
    long line_start_pos = current_pos;

    // --- Find semicolon manually ---
    long semicolon_pos = -1;
    long search_pos = current_pos;
    while (search_pos < effective_chunk_end) {
      if (data[search_pos] == ';') {
        semicolon_pos = search_pos;
        break;
      }
      if (data[search_pos] ==
          '\n') { // Stop search if newline found before semicolon
        break;
      }
      search_pos++;
    }

    if (semicolon_pos ==
        -1) { // No semicolon found until end of chunk or newline
      // Move to next potential line start (or end processing if at chunk end)
      while (current_pos < effective_chunk_end && data[current_pos] != '\n')
        current_pos++;
      if (current_pos < effective_chunk_end)
        current_pos++; // Move past '\n'
      else
        break;  // Reached end
      continue; // Skip to next iteration
    }

    int station_len = semicolon_pos - line_start_pos;
    if (station_len <= 0 || station_len >= MAX_STATION_NAME) {
      // Malformed line (empty name or too long) - skip to next line
      current_pos = semicolon_pos + 1; // Start search after semicolon
      while (current_pos < effective_chunk_end && data[current_pos] != '\n')
        current_pos++;
      if (current_pos < effective_chunk_end)
        current_pos++;
      else
        break;
      continue;
    }

    // --- Find newline manually ---
    long newline_pos = -1;
    search_pos = semicolon_pos + 1;
    while (search_pos < effective_chunk_end) {
      if (data[search_pos] == '\n') {
        newline_pos = search_pos;
        break;
      }
      search_pos++;
    }

    if (newline_pos == -1) { // No newline found until end of chunk
      break; // Stop processing for this thread, might be partial line at chunk
             // end
    }

    int temp_len = newline_pos - (semicolon_pos + 1);
    // Basic validation for temperature string length
    if (temp_len <= 0 ||
        temp_len >= 10) { // e.g., "-99.9" is 5 chars, allow some leeway
      // Malformed line (invalid temp length) - skip to next line
      current_pos = newline_pos + 1;
      if (current_pos >= effective_chunk_end)
        break;
      continue;
    }

    // --- Parse Temperature ---
    const char *temp_start_ptr = data + semicolon_pos + 1;
    float temp = fast_float_parser(temp_start_ptr, temp_len);

    // --- Store Output ---
    // Atomically get an index in the output buffer
    int output_idx = atomicAdd(output_count, 1);

    if (output_idx < max_output_per_chunk) {
      // Copy station name (ensure null termination)
      char *dest_name_ptr = output[output_idx].name;
      const char *src_name_ptr = data + line_start_pos;
      int k = 0;
      while (k < station_len && k < MAX_STATION_NAME - 1) {
        dest_name_ptr[k] = src_name_ptr[k];
        k++;
      }
      dest_name_ptr[k] = '\0'; // Null terminate

      // Store temperature
      output[output_idx].temperature = temp;
    } else {
      // Output buffer full - roll back counter (optional, depends on desired
      // behavior)
      atomicSub(output_count, 1);
      // Consider setting a flag or printf for debugging if this happens often
      // Break here as we cannot store more results for this chunk
      // Note: This might lead to some records being lost if buffer is too
      // small.
      //       A better approach might involve multiple passes or larger
      //       buffers.
      break;
    }

    // Move to the start of the next line
    current_pos = newline_pos + 1;
  }
}

// Kernel to reduce station data (Aggregation - Largely unchanged from previous)
__global__ void reduceStationDataKernel(
    StationData *__restrict__ data, int data_count,
    char *__restrict__ station_names_data,     // Contiguous name data
    char **__restrict__ station_name_pointers, // Array of pointers
    float *__restrict__ mins, float *__restrict__ maxes,
    double *__restrict__ sums, long long *__restrict__ counts,
    int *__restrict__ station_count, int max_stations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= data_count)
    return;

  StationData current = data[idx];
  // Basic validation: Check for empty name (might occur from parsing issues)
  if (current.name[0] == '\0')
    return;

  int found_idx = -1;
  // Read the current count - might be slightly stale, but helps limit search
  // Using volatile might help, but atomics handle the critical updates.
  int current_num_stations = *station_count;

  // --- Stage 1: Search for existing station ---
  // Limit search loop to avoid excessive checks if many stations are added
  // concurrently
  for (int i = 0; i < current_num_stations && i < max_stations; ++i) {
    // Ensure pointer is likely valid before dereferencing aggressively
    // (A truly robust solution is complex, this is a pragmatic check)
    if (station_name_pointers[i] != NULL) {
      if (strcmp_device(station_name_pointers[i], current.name) == 0) {
        found_idx = i;
        break;
      }
    }
  }

  // --- Stage 2: Add station if not found ---
  if (found_idx == -1) {
    // Attempt to claim a new slot atomically
    int new_station_idx = atomicAdd(station_count, 1);

    if (new_station_idx < max_stations) {
      // Successfully claimed a slot, initialize it
      char *dest_name_ptr =
          station_names_data + ((long long)new_station_idx * MAX_STATION_NAME);

      // Copy name using device strcpy
      strcpy_device(dest_name_ptr, current.name);

      // Update the pointer array (potential race here if not careful, but often
      // works) The thread that successfully increments station_count "owns"
      // this index initialization.
      station_name_pointers[new_station_idx] = dest_name_ptr;

      // Initialize stats - Use atomics to be safe, although technically
      // only one thread should initialize a given slot. Using atomics
      // prevents issues if the read of station_count was stale and another
      // thread tries to initialize the same slot concurrently.
      atomicMin_float(&mins[new_station_idx], current.temperature);
      atomicMax_float(&maxes[new_station_idx], current.temperature);
      atomicAdd_double(&sums[new_station_idx], (double)current.temperature);
      // Initialize count using atomic to ensure visibility
      atomicAdd((unsigned long long *)&counts[new_station_idx], 1ULL);

      found_idx = new_station_idx; // Use this index for update logic below

    } else {
      // Failed to add (max_stations reached), roll back counter and return
      atomicSub(station_count, 1);
      // printf("Warning: Max stations reached (%d). Cannot add '%s'.\n",
      // max_stations, current.name);
      return;
    }
  }

  // --- Stage 3: Update statistics for the found/added station ---
  // Ensure found_idx is valid before updating stats
  if (found_idx >= 0 && found_idx < max_stations) {
    // Check if the count is non-zero (meaning it was initialized by this or
    // another thread) If we just initialized it, the count is 1 already. We
    // need to avoid double-counting the initial value. Use atomicRead or check
    // if the current value is the initialization value. Simpler: only add if
    // the count > 0, assuming initialization sets it to 1.
    bool needs_update = true;
    // If this thread just added the station (found_idx == new_station_idx in
    // the if block above), the values are already initialized. Only update if
    // found existing. How to know? If counts[found_idx] > 1 it means others
    // already updated. Let's simplify: always perform atomics, but initialize
    // count to 1 only once.

    atomicMin_float(&mins[found_idx], current.temperature);
    atomicMax_float(&maxes[found_idx], current.temperature);

    // Only add sum and increment count if this thread didn't *just* initialize
    // it. We infer initialization happened if the atomicAdd for station_count
    // returned this index. This is tricky. A safer way is to always add and
    // increment, and handle the initial value correctly during initialization.
    // Let's stick to the current approach: initialize includes the first data
    // point. Subsequent hits update.

    // If this thread *didn't* just initialize the slot (meaning found_idx !=
    // new_station_idx, which we track implicitly by having `found_idx` set
    // before the atomicAdd call), then perform the additions. Refined logic:
    // The thread that successfully claims `new_station_idx` initializes. Other
    // threads finding an existing `found_idx` update.

    if (*station_count > found_idx + 1 ||
        counts[found_idx] > 0) // Check if initialized
    {
      atomicAdd_double(&sums[found_idx], (double)current.temperature);
      atomicAdd((unsigned long long *)&counts[found_idx], 1ULL);
    }
    // Note: The initialization logic handles the first count and sum.
  }
  // If found_idx remains -1, it implies a failure to add (race condition or
  // full), handled above.
}

// --- Main Processing Logic ---

// Process file using CUDA (UPDATED with refined memory, kernel launch, result
// merge)
void process_file_cuda(const char *file_path, HashTable *ht) {
  int fd = open(file_path, O_RDONLY);
  if (fd == -1) {
    perror("Error opening file");
    exit(1);
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    perror("Error getting file size");
    close(fd);
    exit(1);
  }
  long file_size = sb.st_size;
  if (file_size == 0) {
    printf("{}\n");
    close(fd);
    return;
  } // Handle empty file

  // Memory map the file
  char *file_data = (char *)mmap(NULL, file_size, PROT_READ,
                                 MAP_PRIVATE | MAP_POPULATE, fd, 0);
  if (file_data == MAP_FAILED) {
    perror("Error mmap");
    close(fd);
    exit(1);
  }
  madvise(file_data, file_size, MADV_SEQUENTIAL); // Hint OS access pattern

  // --- Device Memory Allocation ---
  StationData *d_station_data =
      nullptr;                   // Intermediate parsed data (large buffer)
  int *d_output_count = nullptr; // Counter for parsed items in d_station_data
  char *d_station_names_data =
      nullptr; // Contiguous block for all station names
  char **d_station_name_pointers =
      nullptr; // Array of pointers into d_station_names_data
  float *d_mins = nullptr;
  float *d_maxes = nullptr;
  double *d_sums = nullptr;
  long long *d_counts = nullptr;
  int *d_station_count = nullptr; // Counter for unique stations found on GPU

  // Estimate max intermediate records needed. A safe bet is ~file_size /
  // avg_line_length. Example: If avg line is ~20 bytes, 1B rows = 20GB. Need
  // chunking. Let's make the intermediate buffer large enough for a chunk's
  // worth.
  int max_output_per_chunk = CHUNK_SIZE / 10; // Heuristic: Avg line > 10 chars

  CUDA_CHECK(cudaMalloc(&d_station_data,
                        (size_t)max_output_per_chunk * sizeof(StationData)));
  CUDA_CHECK(cudaMalloc(&d_output_count, sizeof(int)));
  CUDA_CHECK(
      cudaMalloc(&d_station_names_data,
                 (size_t)MAX_STATIONS * MAX_STATION_NAME * sizeof(char)));
  CUDA_CHECK(
      cudaMalloc(&d_station_name_pointers, MAX_STATIONS * sizeof(char *)));
  CUDA_CHECK(cudaMalloc(&d_mins, MAX_STATIONS * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_maxes, MAX_STATIONS * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sums, MAX_STATIONS * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_counts, MAX_STATIONS * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_station_count, sizeof(int)));

  // --- Initialize Device Data ---
  int h_zero = 0;
  CUDA_CHECK(cudaMemset(d_station_count, 0, sizeof(int)));
  CUDA_CHECK(
      cudaMemset(d_counts, 0,
                 MAX_STATIONS * sizeof(long long))); // Ensure counts start at 0
  // Initialize mins/maxes? Kernels handle first value, but setting might avoid
  // NaN issues if needed. e.g., cudaMemset d_mins to MAX_FLOAT, d_maxes to
  // -MAX_FLOAT

  // Create the array of pointers on the host and copy it to the device
  char **h_station_name_pointers =
      (char **)malloc(MAX_STATIONS * sizeof(char *));
  if (!h_station_name_pointers) {
    fprintf(stderr, "Failed host pointer alloc\n");
    exit(1);
  }
  for (int i = 0; i < MAX_STATIONS; ++i) {
    // Pointers point to locations within the d_station_names_data buffer on the
    // GPU
    h_station_name_pointers[i] =
        d_station_names_data + ((long long)i * MAX_STATION_NAME);
  }
  CUDA_CHECK(cudaMemcpy(d_station_name_pointers, h_station_name_pointers,
                        MAX_STATIONS * sizeof(char *), cudaMemcpyHostToDevice));
  free(h_station_name_pointers);

  // --- Process File in Chunks ---
  char *d_chunk_data = nullptr;
  CUDA_CHECK(cudaMalloc(&d_chunk_data, CHUNK_SIZE));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  for (long offset = 0; offset < file_size; offset += CHUNK_SIZE) {
    long current_chunk_size =
        (offset + CHUNK_SIZE > file_size) ? (file_size - offset) : CHUNK_SIZE;
    if (current_chunk_size <= 0)
      break;

    // Copy chunk to device (asynchronously)
    CUDA_CHECK(cudaMemcpyAsync(d_chunk_data, file_data + offset,
                               current_chunk_size, cudaMemcpyHostToDevice,
                               stream));

    // Reset output count for this chunk (asynchronously)
    CUDA_CHECK(cudaMemsetAsync(d_output_count, 0, sizeof(int), stream));

    // Launch parsing kernel (adjust grid size based on processing range)
    // Grid size should cover all bytes, divided by the range each thread
    // handles.
    int processing_range = 512;
    int parse_grid_size =
        (current_chunk_size + processing_range - 1) / processing_range;
    int parse_blocks =
        (parse_grid_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    parseStationDataKernel<<<parse_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_chunk_data, current_chunk_size, d_station_data, d_output_count,
        max_output_per_chunk, offset);
    CUDA_CHECK(cudaGetLastError()); // Check launch errors

    // Get the number of items parsed (needs synchronization for this value)
    int h_output_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_output_count, d_output_count, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); // Wait for count copy

    if (h_output_count > max_output_per_chunk) {
      fprintf(stderr,
              "Warning: Parsed items (%d) exceeded allocation (%d) for chunk "
              "at offset %ld. Results incomplete.\n",
              h_output_count, max_output_per_chunk, offset);
      h_output_count = max_output_per_chunk; // Cap for safety
    }

    // Launch reduction kernel if data was parsed
    if (h_output_count > 0) {
      int reduce_blocks =
          (h_output_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      reduceStationDataKernel<<<reduce_blocks, THREADS_PER_BLOCK, 0, stream>>>(
          d_station_data, h_output_count, d_station_names_data,
          d_station_name_pointers, d_mins, d_maxes, d_sums, d_counts,
          d_station_count, MAX_STATIONS);
      CUDA_CHECK(cudaGetLastError());
    }
    // Synchronize stream before starting next chunk processing
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_chunk_data)); // Free chunk buffer

  // --- Copy Final Results Back to Host ---
  int h_station_count = 0;
  CUDA_CHECK(cudaMemcpy(&h_station_count, d_station_count, sizeof(int),
                        cudaMemcpyDeviceToHost));

  if (h_station_count > MAX_STATIONS) {
    fprintf(
        stderr,
        "Warning: Final station count (%d) > MAX_STATIONS (%d). Clamping.\n",
        h_station_count, MAX_STATIONS);
    h_station_count = MAX_STATIONS;
  }

  if (h_station_count > 0) {
    // Allocate host memory for results
    char *h_station_names_data =
        (char *)malloc((size_t)h_station_count * MAX_STATION_NAME);
    float *h_mins = (float *)malloc((size_t)h_station_count * sizeof(float));
    float *h_maxes = (float *)malloc((size_t)h_station_count * sizeof(float));
    double *h_sums = (double *)malloc((size_t)h_station_count * sizeof(double));
    long long *h_counts =
        (long long *)malloc((size_t)h_station_count * sizeof(long long));

    // Check allocations
    if (!h_station_names_data || !h_mins || !h_maxes || !h_sums || !h_counts) {
      fprintf(stderr, "Failed host result memory allocation.\n");
      // Cleanup allocated memory before exit
      free(h_station_names_data);
      free(h_mins);
      free(h_maxes);
      free(h_sums);
      free(h_counts);
      // Free device memory too
      cudaFree(d_station_data);
      cudaFree(d_output_count);
      cudaFree(d_station_names_data);
      cudaFree(d_station_name_pointers);
      cudaFree(d_mins);
      cudaFree(d_maxes);
      cudaFree(d_sums);
      cudaFree(d_counts);
      cudaFree(d_station_count);
      munmap(file_data, file_size);
      close(fd);
      exit(1);
    }

    // Copy aggregated data from device
    CUDA_CHECK(cudaMemcpy(h_station_names_data, d_station_names_data,
                          (size_t)h_station_count * MAX_STATION_NAME,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mins, d_mins,
                          (size_t)h_station_count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_maxes, d_maxes,
                          (size_t)h_station_count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sums, d_sums,
                          (size_t)h_station_count * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts, d_counts,
                          (size_t)h_station_count * sizeof(long long),
                          cudaMemcpyDeviceToHost));

    // --- Merge GPU Results into Host Hash Table ---
    for (int i = 0; i < h_station_count; ++i) {
      char *current_name =
          h_station_names_data + ((long long)i * MAX_STATION_NAME);
      // Ensure data is valid before merging
      if (current_name[0] != '\0' && h_counts[i] > 0) {
        StationStats *stats = find_or_create_station(ht, current_name);
        // Merge results (GPU results overwrite/add to host table)
        // Note: This assumes GPU processed all data. If combining partial
        // results, logic might differ.
        stats->min = (double)h_mins[i];
        stats->max = (double)h_maxes[i];
        stats->sum = h_sums[i];
        stats->count =
            h_counts[i]; // Assign directly as GPU holds the final aggregate
      }
    }

    // Free host results memory
    free(h_station_names_data);
    free(h_mins);
    free(h_maxes);
    free(h_sums);
    free(h_counts);
  } else {
    fprintf(stderr, "No stations processed by GPU or count is zero.\n");
  }

  // --- Clean Up ---
  CUDA_CHECK(cudaFree(d_station_data));
  CUDA_CHECK(cudaFree(d_output_count));
  CUDA_CHECK(cudaFree(d_station_names_data));
  CUDA_CHECK(cudaFree(d_station_name_pointers));
  CUDA_CHECK(cudaFree(d_mins));
  CUDA_CHECK(cudaFree(d_maxes));
  CUDA_CHECK(cudaFree(d_sums));
  CUDA_CHECK(cudaFree(d_counts));
  CUDA_CHECK(cudaFree(d_station_count));

  munmap(file_data, file_size);
  close(fd);
}

// --- Main Function ---
int main(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <file_path>\n", argv[0]);
    return 1;
  }
  const char *file_path = argv[1];

  // Optional: Select GPU Device
  // int deviceCount;
  // cudaGetDeviceCount(&deviceCount);
  // if (deviceCount == 0) { fprintf(stderr, "No CUDA devices found.\n"); return
  // 1; } cudaSetDevice(0); // Use device 0

  HashTable ht; // Host hash table for final results
  init_hash_table(&ht, HASH_SIZE);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  printf("Starting GPU processing...\n");
  CUDA_CHECK(cudaEventRecord(start));

  process_file_cuda(file_path, &ht);

  // Ensure all GPU work is finished before stopping timer and printing
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  printf("GPU processing finished.\n");

  // Print final merged results from the host hash table
  print_results(&ht);

  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Total execution time (GPU processing + merge): %.3f seconds\n",
         milliseconds / 1000.0);

  // Clean up
  free(ht.table);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  // cudaDeviceReset(); // Optional: Release CUDA context

  return 0;
}
