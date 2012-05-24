all: *.go 
	go install
	gofmt -w *.go

test: *.go
	go test

bench: *.go
	go test -bench=.

opt: *.go
	go tool 6g -m *.go | go-optview -w
	gofmt -w *.go

.PHONY: clean
clean: *.go
	go tool 6g -m *.go | go-optview -c -w
	gofmt -w *.go
