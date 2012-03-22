all: *.go 
	make -C mx
	go tool vet *.go
	gofmt -w *.go
	go install
	dot -Tpng -O whiteboard.dot
	ln -sf $(CURDIR)/pre-commit .git/hooks/pre-commit

test:
	make test -C mx
