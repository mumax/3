all:
	make -C daemon
	make -C submit
	make doc

clean:
	make clean -C daemon
	make clean -C submit
	rm -rf *.out

doc:
	godoc github.com/barnex/mumax-daemon > README

