all: *.go 
	go install -v
	gofmt -w *.go

test: *.go
	go test

bench: *.go
	go test -bench=.
