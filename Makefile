all: githook 6g tools

PREFIX=code.google.com/p/nimble-cube

PKGS=\
	$(PREFIX)/gpu/conv\
	$(PREFIX)/gpu\
	$(PREFIX)/cpu\
	$(PREFIX)/uni\
	$(PREFIX)/mag\
	$(PREFIX)/dump\
	$(PREFIX)/nimble\
	$(PREFIX)/core\

6g: ptx
	go install -v $(PKGS)
	go install -v 

tools:
	make -C tools 

GCCGO=gccgo -gccgoflags '-static-libgcc -O4 -Ofast -march=native'

gccgo: ptx
	go install -v -compiler $(GCCGO) $(PKGS)
	go install -v -compiler $(GCCGO)

ptx:
	make -C gpu/ptx -j8

githook:
	ln -sf $(CURDIR)/pre-commit .git/hooks/pre-commit
	ln -sf $(CURDIR)/post-commit .git/hooks/post-commit

test: 6gtest  unittest gccgotest

unittest:
	make -C test

6gtest: 6g
	go test -i $(PKGS) 

gccgotest: gccgo
	go test -i -compiler=$(gccgo) $(PKGS)

.PHONY: clean
clean:
	go clean -i -x $(PKGS)
	rm -rf $(GOPATH)/pkg/gccgo/$(PREFIX)/
	make clean -C gpu/ptx
