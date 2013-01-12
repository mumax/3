CUFILES = $(wildcard *.cu)
PTXFILES = $(CUFILES:.cu=.ptx)
GOFILES=$(CUFILES:.cu=.go)

NVCC = nvcc\
	-gencode arch=compute_30,code=sm_30\
	--compiler-options -Werror\
	--compiler-options -Wall\
	-Xptxas\
	-O3\

all: $(GOFILES) install cuda2go

cuda2go: cuda2go.go
	go build cuda2go.go

install: $(GOFILES)
	go install -v

$(GOFILES): %.go: %.ptx cuda2go
	./cuda2go $(@:.go=.cu) 
	gofmt -w $@

$(PTXFILES): %.ptx: %.cu
	rm -f $@
	$(NVCC) -ptx $(@:.ptx=.cu)

.PHONY: clean
clean:
	rm -f $(PTXFILES)
	rm -f $(GOFILES)

