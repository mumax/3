CUFILES = $(wildcard *.cu)
PTXFILES = $(CUFILES:.cu=.ptx)
GOFILES=$(CUFILES:.cu=.go)

NVCC = nvcc -G

all: $(GOFILES) install

install: $(GOFILES)
	go install -v

$(GOFILES): %.go: %.ptx
	./ptx2go $(@:.go=.ptx) $@ $(@:.go=)

$(PTXFILES): %.ptx: %.cu
	rm -f $@
	$(NVCC) -ptx $(@:.ptx=.cu)

.PHONY: clean
clean:
	rm -f $(PTXFILES)
	rm -f $(GOFILES)

