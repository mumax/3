package main

// Output for gnuplot's "splot"
// Author: Arne Vansteenkiste

import (
	"bufio"
	"fmt"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"io"
)

func dumpGnuplot(out io.Writer, f *data.Slice) (err error) {
	buf := bufio.NewWriter(out)
	defer buf.Flush()

	data := f.Tensors()
	gridsize := f.Mesh().Size()
	cellsize := f.Mesh().CellSize()
	// If no cell size is set, use generic cell index.
	if cellsize == [3]float64{0, 0, 0} {
		cellsize = [3]float64{1, 1, 1}
	}
	ncomp := f.NComp()

	// Here we loop over X,Y,Z, not Z,Y,X, because
	// internal in C-order == external in Fortran-order
	for i := 0; i < gridsize[0]; i++ {
		x := float64(i) * cellsize[0]
		for j := 0; j < gridsize[1]; j++ {
			y := float64(j) * cellsize[1]
			for k := 0; k < gridsize[2]; k++ {
				z := float64(k) * cellsize[2]
				_, err = fmt.Fprint(buf, z, " ", y, " ", x, "\t")
				for c := 0; c < ncomp; c++ {
					_, err = fmt.Fprint(buf, data[util.SwapIndex(c, ncomp)][i][j][k], " ") // converts to user space.
				}
				_, err = fmt.Fprint(buf, "\n")
			}
			_, err = fmt.Fprint(buf, "\n")
		}
	}
	return
}
