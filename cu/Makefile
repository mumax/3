NVCC=nvcc -g 

cufiles=$(wildcard *.cu)
ptxfiles=$(cufiles:.cu=.ptx)

all: $(ptxfiles)

$(ptxfiles): %.ptx: %.cu %.h
	rm -f $@
	$(NVCC) -ptx $(@:.ptx=.cu)

.PHONY: clean
clean:
	rm -f *.ptx

