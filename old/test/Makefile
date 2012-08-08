GOFILES = $(wildcard *.go)

all: $(GOFILES)

.PHONY: $(GOFILES)
$(GOFILES):
	go run $(@)

.PHONY: clean
clean:
	rm -f $(GOFILES:.go=)
	rm -f $(GOFILES:.go=.dot)
	rm -f $(GOFILES:.go=.dot.pdf)
