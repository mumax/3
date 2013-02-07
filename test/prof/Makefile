MEM = $(wildcard mem*.go)

all: $(MEM)

.PHONY: $(MEM)

$(MEM):
	go run $@ -memprof -o $(@:.go=.out)

