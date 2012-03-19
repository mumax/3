all: *.go 
	go tool vet *.go
	gofmt -w *.go
	go install

test: *.go
	go test
