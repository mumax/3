all: 6g doc

6g:
	go install -v
	go tool vet *.go
	gofmt -w *.go

GCCGO=gccgo -gccgoflags '-static-libgcc -O3'

gccgo:
	go install -v -compiler $(GCCGO)

test: 6gtest gccgotest

6gtest: 
	go test

gccgotest: 
	go test -compiler $(GCCGO)

bench: 6gbench gccgobench

6gbench:
	go test -bench=.

gccgobench:
	go test -bench=. -compiler $(GCCGO)

clean:
	go clean
	go-optview -c -w *.go
	gofmt -w *.go

opt:
	go-optview -w *.go
	gofmt -w *.go

doc:
	godoc github.com/barnex/cuda4 > README
