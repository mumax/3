all:
	goimports -w *.go
	./make.bash
	go fmt 
	go build -v 
	go build test.go
