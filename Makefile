# Compiler
CC = mpicc

# Compiler flags
CFLAGS = -O3 -fopenmp -Wall -Wextra -std=c99

# Executable name
EXEC = 1brc-hybrid

# Source files
SRC = 1brc-hybrid.c

# Object files
OBJ = $(SRC:.c=.o)

# Default target
all: $(EXEC)

# Build the executable
$(EXEC): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

# Compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJ) $(EXEC)

# Run the program (example usage)
run:
	mpirun -np 4 ./$(EXEC) input.txt

.PHONY: all clean run
