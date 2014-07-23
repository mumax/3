all:
	gofmt -w *.go > /dev/null 2> /dev/null || echo "" ;
	go install
	go vet
