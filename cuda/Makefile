CUFILES = $(wildcard *.cu)
GOFILES=$(CUFILES:.cu=_wrapper.go)

NVCC = nvcc\
	-Xptxas\
	--compiler-options -Werror\
	--compiler-options -Wall\
	-O3\

all: $(GOFILES)
	go fmt	
	go install -v

$(GOFILES): %.go: $(@:_wrapper.go=.cu)
	$(NVCC) -gencode arch=compute_30,code=sm_30 $(@:_wrapper.go=.cu) -o $(@:_wrapper.go=_30.ptx)
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

