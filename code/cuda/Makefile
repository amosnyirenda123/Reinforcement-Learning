# ============================================================
# CUDA Makefile
# Arch Linux + NVIDIA CUDA Toolkit
# ============================================================

# Arch Linux CUDA installation
CUDA_PATH     ?= /opt/cuda-12.6
HOST_COMPILER ?= /usr/bin/g++-13

NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# Include current directory
INCLUDES := -I.

# Compiler flags
NVCC_FLAGS := -m64 -lineinfo -std=c++17

# Build directory
BUILD_DIR := build

# GPU architecture(s) to build for.
# Quadro P2000 = Pascal = compute capability 6.1 (sm_61).
# NOTE: this only works because CUDA_PATH points at 12.6 -
# CUDA 13.0+ dropped Pascal/Maxwell/Volta entirely, so sm_61
# would fail to compile against /opt/cuda (13.3).
SMS ?= 61
GENCODE_FLAGS := $(foreach sm,$(SMS),-gencode arch=compute_$(sm),code=sm_$(sm))

SRCS    ?= $(wildcard *.cu)
TARGETS := $(patsubst %.cu,$(BUILD_DIR)/%,$(SRCS))


# ============================================================
# Targets
# ============================================================

.PHONY: all clean run list help

# Build all .cu files
all: $(TARGETS)

# Compile .cu -> executable
$(BUILD_DIR)/%: %.cu | $(BUILD_DIR)
	$(NVCC) $(INCLUDES) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $<

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build and run one program
#
# Example:
#   make run FILE=device_info
#
run:
	@file="$(basename $(FILE))"; \
	$(MAKE) $(BUILD_DIR)/$$file; \
	./$(BUILD_DIR)/$$file

# Remove build directory
clean:
	rm -rf $(BUILD_DIR)

# List discovered CUDA files
list:
	@echo "Discovered .cu files:"
	@for f in $(SRCS); do echo "  $$f"; done

# Show help
help:
	@echo "make"
	@echo "    Build all .cu files"
	@echo
	@echo "make SRCS=\"a.cu b.cu\""
	@echo "    Build only selected files"
	@echo
	@echo "make run FILE=device_info"
	@echo "    Build and run device_info.cu"
	@echo
	@echo "make clean"
	@echo "    Remove build/ directory"
	@echo
	@echo "make list"
	@echo "    List discovered .cu files"