package main

// Output for gnuplot's "splot"

import (
	"bufio"
	"fmt"
	"io"

	"github.com/mumax/3/v3/data"
)

const DELIM = "\t"

func dumpGnuplot(f *data.Slice, m data.Meta, out io.Writer) {
	buf := bufio.NewWriter(out)
	defer buf.Flush()

	data := f.Tensors()
	cellsize := m.CellSize
	// If no cell size is set, use generic cell index.
	if cellsize == [3]float64{0, 0, 0} {
		cellsize = [3]float64{1, 1, 1}
	}
	ncomp := f.NComp()

	for iz := range data[0] {
		z := float64(iz) * cellsize[Z]
		for iy := range data[0][iz] {
			y := float64(iy) * cellsize[Y]
			for ix := range data[0][iz][iy] {
				x := float64(ix) * cellsize[X]
				fmt.Fprint(buf, x, DELIM, y, DELIM, z, DELIM)
				for c := 0; c < ncomp-1; c++ {
					fmt.Fprint(buf, data[c][iz][iy][ix], DELIM)
				}
				fmt.Fprint(buf, data[ncomp-1][iz][iy][ix])
				fmt.Fprint(buf, "\n")
			}
			fmt.Fprint(buf, "\n")
		}
	}
}
