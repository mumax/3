dotfiles=$(wildcard *.dot)
pdffiles=$(dotfiles:.dot=.pdf)

all:$(pdffiles)

$(pdffiles): %.pdf: %.dot
	dot -Tpdf $(@:.pdf=.dot) > $@

.PHONY: clean
clean:
	rm -f $(pdffiles)
