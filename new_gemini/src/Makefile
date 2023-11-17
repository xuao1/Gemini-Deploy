CUDA_PATH ?= /usr/local/cuda-11.7
PWD := $(shell pwd)
PREFIX ?= $(PWD)/..

SMS ?= 80
GENCODE_FLAGS += $(foreach sm,$(SMS),-gencode arch=compute_$(sm),code=sm_$(sm))

CXX ?= g++-8
NVCC ?= $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

CUDA_LDFLAGS += -lcuda -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib64/stubs
LDFLAGS += -ldl -lrt

CXXFLAGS += -std=c++11 -fPIC

ifeq ($(DEBUG),1)
CXXFLAGS += -g -D_DEBUG -Wall
else
CXXFLAGS += -O2
endif

# Target rules
all: libgemhook.so.1

debug.o: debug.cpp debug.h
	g++ -fPIC $(CXXFLAGS) -o $@ -c $<

hook.o: hook.cpp debug.h
	$(NVCC) -m64 --compiler-options "$(CXXFLAGS)" $(GENCODE_FLAGS) -o $@ -c $<


libgemhook.so.1: hook.o debug.o
	$(EXEC) $(NVCC) -shared -m64 $(GENCODE_FLAGS) -o $@ $+ $(CUDA_LDFLAGS) $(LDFLAGS)
	$(EXEC) mkdir -p $(PREFIX)/lib
	$(EXEC) cp $@ $(PREFIX)/lib

clean:
	rm -f *.o
