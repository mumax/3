all:
	(goimports -w *.go > /dev/null 2> /dev/null) || echo ""
	(gofmt -w *.go > /dev/null 2> /dev/null ) || echo ""
	go install
