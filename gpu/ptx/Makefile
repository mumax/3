CUFILES = $(wildcard *.cu)
PTXFILES = $(CUFILES:.cu=.ptx)
GOFILES=$(CUFILES:.cu=.go)

NVCC = nvcc\
	-gencode arch=compute_30,code=sm_30\
	--compiler-options -Werror\
	--compiler-options -Wall\
	-Xptxas\
	-O3\
	-v\

all: $(GOFILES) install

install: $(GOFILES)
	go install -v

$(GOFILES): %.go: %.ptx
	./ptx2go $(@:.go=.ptx) $@ $(@:.go=)

$(PTXFILES): %.ptx: %.cu *.h
	rm -f $@
	$(NVCC) -ptx $(@:.ptx=.cu)

.PHONY: clean
clean:
	rm -f $(PTXFILES)
	rm -f $(GOFILES)

