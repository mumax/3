SRC = $(filter-out doc.go, $(wildcard *.go))
OUT = $(SRC:.go=)

all: $(OUT)

.PHONY: $(OUT)

$(OUT): %: %.go
	go run $@.go -f -s -o $@.out

.PHONY: clean

clean:
	rm -rf *.out

