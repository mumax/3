all: 6g #gccgo vet

6g:
	go install -v
	gofmt -w *.go

vet:
	go tool vet *.go

GCCGO=gccgo -gccgoflags '-static-libgcc -O3 -g'

gccgo:
	go build -v -compiler $(GCCGO)

test: 6gtest gccgotest

6gtest: 
	go test -i
	go test

gccgotest: 
	go test -i -compiler $(GCCGO)
	go test -compiler $(GCCGO)

bench: 6gbench #gccgobench

6gbench:
	go test -bench=.

gccgobench:
	go test -compiler $(GCCGO) -bench=. 

clean:
	go-optview -w -c *.go
	gofmt -w *.go	
	go clean
	rm -f *.{5,6,8}

opt:
	go-optview -w *.go
	gofmt -w *.go
