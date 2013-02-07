SRC = $(filter-out doc.go, $(wildcard *.go))
OUT = $(SRC:.go=)

all: $(OUT)

.PHONY: $(OUT)

$(OUT): %: %.go
	go run $@.go -s -o $@.out

.PHONY: clean

clean:
	rm -rf *.out

