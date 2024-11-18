BUILDDIR="build"
STATIC="static"
BENCH="../bench"

# build the html pages in ${BUILDDIR}
.PHONY: html
html: doc mumax3libs gpus
	mkdir -p ${BUILDDIR}
	./doc -examples -builddir ${BUILDDIR}
	cp ${STATIC}/* build

.PHONY: doc
doc:
	go build -v

.PHONY: mumax3libs
mumax3libs:
	go install -v github.com/mumax/3/cmd/...

.PHONY: gpus
gpus:
	# Only attempt move when gnuplot succeeded
	if (cd ${BENCH} && gnuplot gpus.gplot); then \
		mv ${BENCH}/gpus.svg ${STATIC}/; \
	fi

.PHONY: clean
clean:
	rm -rf build
	rm -f doc