quick:
	boilerplate *.t

all:
	./latex2png llg lltorque bexch bexch2
	./precession.gplot
	./heun.gplot
	mumax3-convert -gplot -resize 128x1x1 exchange1d.out/m000002.ovf
	./exchange1d.gplot
	boilerplate *.t
