all: githook gofmt 6g #gccgo

PKGS=\
	core

6g:
	go install -v nimble-cube/$(PKGS)

gofmt:
	gofmt -w $(PKGS)/*.go

GCCGO=gccgo -gccgoflags '-static-libgcc -O3'

gccgo:
	go build -v -compiler $(GCCGO) nimble-cube/$(PKGS)

githook:
	ln -sf $(CURDIR)/pre-commit .git/hooks/pre-commit

test: 6gtest gccgotest

6gtest:
	go test nimble-cube/$(PKGS)

gccgotest:
	go test -compiler $(GCCGO) nimble-cube/$(PKGS)

