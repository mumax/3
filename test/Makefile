SRC = $(wildcard *.go)
OUT = $(SRC:.go=)

all: $(OUT)

.PHONY: $(OUT)

$(OUT): %: %.go
	go run $@.go

