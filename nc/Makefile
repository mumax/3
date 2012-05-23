all: *.go 
	go install
	gofmt -w *.go

test: *.go
	go test

bench: *.go
	go test -bench=.
