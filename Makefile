all: *.go
	go tool vet *.go
	go install
