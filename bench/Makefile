# build the html pages in ${BUILDDIR}
.PHONY: gpus
gpus:
	go build -v
	./bench

.PHONY: clean
clean:
	rm -f bench
	rm gpus_*.txt