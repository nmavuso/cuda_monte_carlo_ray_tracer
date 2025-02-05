# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -O3
CUDA_CXX := nvcc
CUDA_CXXFLAGS := -std=c++17 -O3 -arch=sm_75 # Adjust sm_75 to your GPU architecture

# Directories
BUILD_DIR := build
SRC_DIR := src
INCLUDE_DIR := include
SCRIPTS_DIR := scripts
MODELS_DIR := models

# Executable name
EXECUTABLE := path_tracer

# Source files
SOURCES := $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SOURCES:.cu=.o))
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(OBJECTS:.cpp=.o))

# Header files
HEADERS := $(wildcard $(INCLUDE_DIR)/*.h)

# Default target
all: $(BUILD_DIR)/$(EXECUTABLE)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS) | $(BUILD_DIR)
	$(CUDA_CXX) $(CUDA_CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Compile C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Link object files
$(BUILD_DIR)/$(EXECUTABLE): $(OBJECTS)
	$(CUDA_CXX) $(CUDA_CXXFLAGS) $^ -o $@ -lcudart # Add other libraries if needed

# Train denoiser
train_denoiser: $(SCRIPTS_DIR)/train_denoiser.py
	python3 $< --model_dir $(MODELS_DIR)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean train_denoiser
