GOFILES=$(wildcard *.go)

all: $(GOFILES)

.PHONY: $(GOFILES)
$(GOFILES):
	go build $(@)
	./$(@:.go=.bash)
