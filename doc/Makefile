BUILDDIR="build"
STATIC="static"

# build the html pages in ${BUILDDIR}
.PHONY: html
html: doc mumax3libs
	mkdir -p ${BUILDDIR}
	./doc -examples -builddir ${BUILDDIR}
	cp ${STATIC}/* build

.PHONY: doc
doc:
	go build -v

.PHONY: mumax3libs
mumax3libs:
	go install -v github.com/mumax/3/cmd/...

.PHONY: clean
clean:
	rm -rf build
	rm -f doc