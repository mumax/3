package main

// Output for gnuplot's "splot"
// Author: Arne Vansteenkiste

import (
	"bufio"
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/dump"
	"fmt"
	"os"
)

func dumpGnuplot(f *dump.Frame, file string) {

	out_, err := os.OpenFile(file, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	core.Fatal(err)
	defer out_.Close()

	out_buffered := bufio.NewWriter(out_)
	defer out_buffered.Flush()

	data := f.Tensors()
	gridsize := f.MeshSize
	cellsize := f.MeshStep
	// If no cell size is set, use generic cell index.
	if cellsize == [3]float64{0, 0, 0} {
		cellsize = [3]float64{1, 1, 1}
	}
	ncomp := f.Components
	core.Assert(ncomp > 0)

	// Here we loop over X,Y,Z, not Z,Y,X, because
	// internal in C-order == external in Fortran-order
	for i := 0; i < gridsize[X]; i++ {
		x := float64(i) * cellsize[X]
		for j := 0; j < gridsize[Y]; j++ {
			y := float64(j) * cellsize[Y]
			for k := 0; k < gridsize[Z]; k++ {
				z := float64(k) * cellsize[Z]
				_, err := fmt.Fprint(out_buffered, z, " ", y, " ", x, "\t")
				core.Fatal(err)
				for c := 0; c < ncomp; c++ {
					_, err := fmt.Fprint(out_buffered, data[SwapIndex(c, ncomp)][i][j][k], " ") // converts to user space.
					core.Fatal(err)
				}
				_, err = fmt.Fprint(out_buffered, "\n")
				core.Fatal(err)
			}
			_, err := fmt.Fprint(out_buffered, "\n")
			core.Fatal(err)
		}
		core.Fatal(err)
	}
}
