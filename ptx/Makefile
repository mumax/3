CUFILES = $(wildcard *.cu)
PTXFILES = $(CUFILES:.cu=.ptx)
NVCC = nvcc -G

all: $(PTXFILES)

$(PTXFILES): %.ptx: %.cu
	rm -f $@
	$(NVCC) -ptx $(@:.ptx=.cu)

.PHONY: clean
clean:
	rm -f $(PTXFILES)

