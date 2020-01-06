.PHONY: doc
doc:
	go build -v

.PHONY: html
html: doc
	go install -v github.com/mumax/3/cmd/...
	./doc -vet 2> /dev/null || echo no worries
	./doc

.PHONY: clean
clean:
	rm -rf example*.out example*.mx3
	rm -f doc

.PHONY: realclean
realclean: clean
	rm -f api.html examples.html
