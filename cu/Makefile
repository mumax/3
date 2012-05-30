all: 6g gccgo

LIB=cu

6g:
	go install -v
	go tool vet *.go
	gofmt -w *.go

GCCGO=gccgo -gccgoflags '-static-libgcc -O3'

gccgo:
	go build -v -compiler $(GCCGO)

test: 6gtest gccgotest

6gtest: 
	go test $(LIB)

gccgotest: 
	go test -compiler $(GCCGO) $(LIB)

bench: 6gbench gccgobench

6gbench:
	go test -bench=. $(LIB)

gccgobench:
	go test -bench=. -compiler $(GCCGO) $(LIB)

clean:
	go clean
