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

PKGS=nimble-cube/core nimble-cube/gpu nimble-cube/gpu/conv nimble-cube/dump nimble-cube/mag nimble-cube/unit

6gtest:
	go test $(PKGS) 

gccgotest:
	go test -compiler=$(gccgo) $(PKGS)
