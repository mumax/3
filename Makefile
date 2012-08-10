all: githook gofmt 6g #gccgo

6g:
	go install -v 

gofmt:
	gofmt -w */*.go

GCCGO=gccgo -gccgoflags '-static-libgcc -O3'

gccgo:
	go build -v -compiler $(GCCGO) 

githook:
	ln -sf $(CURDIR)/pre-commit .git/hooks/pre-commit

test: 6gtest gccgotest

6gtest:
	go test 

gccgotest:
	go test -compiler $(GCCGO)

