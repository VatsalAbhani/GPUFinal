NVCC        = nvcc
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 -arch=sm_20
else
NVCC_FLAGS  = -O3 
endif
LD_FLAGS    = -lcudart 
EXE         = ProjectGPU
OBJ         = ProjectGPU.o

default: $(EXE)

ProjectGPU.o: ProjectGPU.cu
	$(NVCC) -c -o $@ ProjectGPU.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
