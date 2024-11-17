# Compiler variables
NVCC = nvcc
CXX = g++
CUDA_LIB = /usr/local/cuda/lib64
CUDA_FLAGS = -lcudart

# File names
TARGET = main
KERNEL_OBJ = kernels.o
KERNEL_SRC = kernels.cu
MAIN_SRC = main.cpp

# Default build target
all: $(TARGET)

# Build the CUDA object file
$(KERNEL_OBJ): $(KERNEL_SRC)
	$(NVCC) -c $(KERNEL_SRC) -o $(KERNEL_OBJ)

# Link the object file with the main C++ program
$(TARGET): $(MAIN_SRC) $(KERNEL_OBJ)
	$(CXX) $(MAIN_SRC) $(KERNEL_OBJ) -o $(TARGET) -L$(CUDA_LIB) $(CUDA_FLAGS)

# Clean up build files
clean:
	rm -f $(KERNEL_OBJ) $(TARGET)

# Phony targets
.PHONY: all clean