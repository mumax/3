MEM = $(wildcard mem*.go)
GPU = $(wildcard gpu*.go)

all: $(MEM) $(GPU)

.PHONY: $(MEM) $(GPU)

$(MEM):
	go run $@ -f -memprof -o $(@:.go=.out)

$(GPU):
	go run $@ -f -gpuprof -o $(@:.go=.out)

.PHONY: clean

clean:
	rm -rf *.out
