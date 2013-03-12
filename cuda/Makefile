CUFILES = $(wildcard *.cu)
GOFILES=$(CUFILES:.cu=_wrapper.go)

NVCC = nvcc\
	--compiler-options -Werror\
	--compiler-options -Wall\
	-Xptxas\
	-O3\
	-ptx

all: $(GOFILES)
	go fmt	
	go install -v

$(GOFILES): %.go: $(@:_wrapper.go=.cu)
	$(NVCC) -gencode arch=compute_20,code=sm_20 $(@:_wrapper.go=.cu) -o $(@:_wrapper.go=_20.ptx)
	$(NVCC) -gencode arch=compute_20,code=sm_21 $(@:_wrapper.go=.cu) -o $(@:_wrapper.go=_21.ptx)
	$(NVCC) -gencode arch=compute_30,code=sm_30 $(@:_wrapper.go=.cu) -o $(@:_wrapper.go=_30.ptx)
	$(NVCC) -gencode arch=compute_35,code=sm_35 $(@:_wrapper.go=.cu) -o $(@:_wrapper.go=_35.ptx)
	touch $@
	

#cuda2go: cuda2go.go
#	go build cuda2go.go
#
#$(GOFILES): %.go: %.ptx cuda2go
#	./cuda2go $(@:.go=.cu) 
#
#$(PTXFILES): %.ptx: %.cu
#	$(NVCC) -ptx $(@:.ptx=.cu)
#
#.PHONY: clean
clean:
	rm -f *.ptx
	rm -f $(GOFILES)
	rm -f cuda2go

