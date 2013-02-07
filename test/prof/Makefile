MEM = $(wildcard mem*.go)

all: $(MEM)

.PHONY: $(MEM)

$(MEM):
	go run $@ -f -memprof -o $(@:.go=.out)

.PHONY: clean

clean:
	rm -rf *.out
