all:
	make -C daemon
	make -C submit

clean:
	make clean -C daemon
	make clean -C submit
	rm -rf *.out
