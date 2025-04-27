# --- Compilers ---
# C Compiler for MPI/OpenMP code
CC = mpicc
# CUDA Compiler (using its own linking is generally recommended for CUDA code)
NVCC = nvcc

# --- Executable Names ---
HYBRID_EXEC = 1brc-hybrid
CUDA_EXEC = 1brc-cuda

# --- Source Files ---
HYBRID_SRC = 1brc-hybrid.c
CUDA_SRC = 1brc-cuda.cu # Assuming your CUDA file is named this

# --- Object Files ---
HYBRID_OBJ = $(HYBRID_SRC:.c=.o)
CUDA_OBJ = $(CUDA_SRC:.cu=.o)

# --- Compiler Flags ---
# Flags for C/MPI/OpenMP code
CFLAGS = -O3 -fopenmp -Wall -Wextra -std=c99
# Flags for CUDA code (adjust std and arch as needed)
CUDA_CFLAGS = -O3 -std=c++11
# Specify GPU architecture (e.g., sm_70 for Volta, sm_80 for Ampere)
# Modify this based on your target GPU!
CUDA_ARCH_FLAGS = -arch=sm_70

# --- Linker Flags ---
# Flags for linking C/MPI/OpenMP code (OpenMP needed at link time too)
LDFLAGS = -fopenmp
# Flags for linking CUDA code (essential CUDA runtime library)
CUDA_LDFLAGS = -lcudart

# --- Targets ---

# Default target: build both executables
all: $(HYBRID_EXEC) $(CUDA_EXEC)

# Build the MPI+OpenMP executable
$(HYBRID_EXEC): $(HYBRID_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build the CUDA executable
$(CUDA_EXEC): $(CUDA_OBJ)
	$(NVCC) $(CUDA_ARCH_FLAGS) -o $@ $^ $(CUDA_LDFLAGS)

# Compile C source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA source files into object files
%.o: %.cu
	$(NVCC) $(CUDA_CFLAGS) $(CUDA_ARCH_FLAGS) -c $< -o $@

# Clean up build files for both executables
clean:
	rm -f $(HYBRID_OBJ) $(HYBRID_EXEC) $(CUDA_OBJ) $(CUDA_EXEC)

# --- Run Targets (Examples) ---

# Run the hybrid program
run-hybrid: $(HYBRID_EXEC)
	mpirun -np 4 ./$(HYBRID_EXEC) input.txt

# Run the CUDA program
run-cuda: $(CUDA_EXEC)
	./$(CUDA_EXEC) input.txt

.PHONY: all clean run-hybrid run-cuda
