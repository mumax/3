all: *.go 
	go install -v
	go tool vet *.go
	gofmt -w *.go
	ln -sf $(CURDIR)/pre-commit .git/hooks/pre-commit

test:
	make test -C nc
	make test -C mm
