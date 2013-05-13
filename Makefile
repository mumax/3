all: nvcc githook 6g mx3 tools

PREFIX=code.google.com/p/mx3

PKGS=\
	$(PREFIX)/engine\
	$(PREFIX)/cuda\
	$(PREFIX)/util\
	$(PREFIX)/draw\
	$(PREFIX)/mag\
	$(PREFIX)/data\
	$(PREFIX)/prof\

$(PREFIX)/cuda: nvcc

.PHONY: mx3
mx3:
	go install -v
	go build -o mx3 main.go

.PHONY: nvcc
nvcc:
	make -C cuda -j8

.PHONY: 6g
6g:
	go install -v $(PKGS)

.PHONY: tools
tools:
	go install -v $(PREFIX)/tools/mx3-convert

GCCGO=gccgo -gccgoflags '-static-libgcc -O4 -Ofast -march=native'

.PHONY: gccgo
gccgo:
	go install -v -compiler $(GCCGO) $(PKGS)
	go install -v -compiler $(GCCGO)

.PHONY: githook
githook:
	ln -sf $(CURDIR)/pre-commit .git/hooks/pre-commit
	ln -sf $(CURDIR)/post-commit .git/hooks/post-commit

.PHONY: racetest
racetest:
	go test -race $(PKGS)
	make -C test

.PHONY: test
test: 6gtest unittest #gccgotest #re-enable gccgotest when gcc up to date with go 1.1

.PHONY: unittest
unittest:
	(cd examples && ./build.bash)
	(cd test && ./run.bash)

.PHONY: 6gtest
6gtest: 6g
	go test -i $(PKGS) 
	go test $(PKGS) 

gccgotest: gccgo
	go test -i -compiler=$(gccgo) $(PKGS)
	go test -compiler=$(gccgo) $(PKGS)

.PHONY: bench
bench: 6g
	go test -test.bench $(PKGS)

.PHONY: gccgobench
gccgobench: gccgo
	go test -compiler=$(gccgo) -test.bench $(PKGS)

.PHONY: clean
clean:
	go clean -i -x $(PKGS)
	rm -rf $(GOPATH)/pkg/gccgo/$(PREFIX)/
	make clean -C cuda

