NVCC=nvcc
cufiles=$(wildcard *.cu)
ptxfiles=$(cufiles:.cu=.ptx)

all: $(ptxfiles)

$(ptxfiles): %.ptx: %.cu
	rm -f $@
	$(NVCC) -ptx $(@:.ptx=.cu)

.PHONY: clean
clean:
	rm -f *.ptx

