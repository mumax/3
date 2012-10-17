all: githook gofmt 6g gccgo

6g: ptx
	go install -v 

gofmt:
	gofmt -w */*.go

GCCGO=gccgo -gccgoflags '-static-libgcc -O3'

gccgo: ptx
	go build -v -compiler $(GCCGO) 

ptx:
	make -C gpu/ptx

githook:
	ln -sf $(CURDIR)/pre-commit .git/hooks/pre-commit
	ln -sf $(CURDIR)/post-commit .git/hooks/post-commit

test: 6gtest gccgotest unittest

unittest:
	make -C test

PKGS=nimble-cube/core nimble-cube/gpu nimble-cube/gpu/conv nimble-cube/dump nimble-cube/mag 

6gtest:
	go test $(PKGS) 

gccgotest:
	go test -compiler=$(gccgo) $(PKGS)
