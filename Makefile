all: githook 6g gccgo

PKGS=\
	nimble-cube/gpu/conv\
	nimble-cube/gpu\
	nimble-cube/mag\
	nimble-cube/dump\
	nimble-cube/core\

6g: ptx
	go install -v $(PKGS)
	go install -v 

GCCGO=gccgo -gccgoflags '-static-libgcc -O3'

gccgo: ptx
	go install -v -compiler $(GCCGO) $(PKGS)
	go install -v -compiler $(GCCGO)

ptx:
	make -C gpu/ptx

githook:
	ln -sf $(CURDIR)/pre-commit .git/hooks/pre-commit
	ln -sf $(CURDIR)/post-commit .git/hooks/post-commit

test: 6gtest gccgotest unittest

unittest:
	make -C test

6gtest:
	go test -i $(PKGS) 

gccgotest:
	go test -i -compiler=$(gccgo) $(PKGS)

.PHONY: clean
clean:
	go clean -i -x $(PKGS)
