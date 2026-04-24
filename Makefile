# ─────────────────────────────────────────────────────────────────────────────
# Makefile — CUDA Batch Image Processor
#
# I target the CUDA Samples Common headers for ImageIO / ImagesCPU / ImagesNPP.
# The CUDA_SAMPLES_PATH variable should point to the root of the CUDA Samples
# repo (or the Common/ subfolder that ships with the CUDA toolkit on the lab).
#
# Typical lab path: /usr/local/cuda/samples
# If yours differs, override on the command line:
#   make CUDA_SAMPLES_PATH=/path/to/cuda-samples
# ─────────────────────────────────────────────────────────────────────────────

CUDA_SAMPLES_PATH ?= /usr/local/cuda/samples

# Where the Common helpers live (ImageIO.h, Exceptions.h, helper_cuda.h …)
COMMON_INC := $(CUDA_SAMPLES_PATH)/Common

# Output binary
BIN_DIR  := bin
TARGET   := $(BIN_DIR)/run

# Source
SRC := src/batchImageProcess.cu

# Compiler
NVCC := nvcc

# Flags
#   -std=c++14     : C++14 minimum for CUDA Samples helpers
#   -O2            : optimize (keeps compile fast enough for dev iteration)
#   -I$(COMMON_INC): find ImageIO.h, Exceptions.h, etc.
#   -lnppif -lnppig -lnppicc -lnppidei -lnppist : NPP image-filter libraries
#   -lnppial -lnppim : additional NPP math / magnitude functions
NVCCFLAGS := -std=c++14 -O2 \
             -I$(COMMON_INC) \
             -lnppif -lnppig -lnppicc -lnppidei \
             -lnppist -lnppial -lnppim

.PHONY: all clean data

all: $(TARGET)

# Create bin/ and compile
$(TARGET): $(SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Generate synthetic sample data (needs Python3 + Pillow)
data:
	python3 scripts/generate_data.py --count 120 --outdir data

# Remove compiled binary
clean:
	rm -rf $(BIN_DIR)
