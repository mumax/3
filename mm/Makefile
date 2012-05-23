all: *.go 
	go install
	go tool vet *.go
	gofmt -w *.go

test: *.go
	go test

bench: *.go
	go test -bench=.
