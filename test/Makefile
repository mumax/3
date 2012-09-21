GOFILES = test4.go#$(wildcard *.go)

all: $(GOFILES)

.PHONY: $(GOFILES)
$(GOFILES):
	go run $(@)

.PHONY: clean
clean:
	rm -f $(GOFILES:.go=)
