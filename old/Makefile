all: 6g gccgo githook

6g:
	go install -v
	go tool vet *.go
	gofmt -w *.go

GCCGO=gccgo -gccgoflags '-static-libgcc -O3'

gccgo:
	go build -v -compiler $(GCCGO)

githook:
	ln -sf $(CURDIR)/pre-commit .git/hooks/pre-commit

test: 6gtest #testdir #gccgotest

6gtest: 
	go test nimble-cube/nc
	go test nimble-cube/mm

gccgotest: 
	go test -compiler $(GCCGO) nimble-cube/nc
	go test -compiler $(GCCGO) nimble-cube/mm

testdir:
	make -C test

bench: 6gbench benchdir #gccgobench 

6gbench:
	go test -bench=. nimble-cube/nc
	go test -bench=. nimble-cube/mm

gccgobench:
	go test -bench=. -compiler $(GCCGO) nimble-cube/nc
	go test -bench=. -compiler $(GCCGO) nimble-cube/mm

benchdir:
	make -C bench

clean:
	go clean
