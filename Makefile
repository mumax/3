all: *.go 
	make -C mx
	go tool vet *.go
	gofmt -w *.go
	go install
	dot -Tpng -O whiteboard.dot
