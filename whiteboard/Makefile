dotfiles=$(wildcard *.dot)
pngfiles=$(dotfiles:.dot=.png)

all:$(pngfiles)

$(pngfiles): %.png: %.dot
	dot -Tpng $(@:.png=.dot) > $@

.PHONY: clean
clean:
	rm -f $(pngfiles)
