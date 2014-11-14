package main

// Output for gnuplot's "splot"

import (
	"bufio"
	"fmt"
	"io"

	"github.com/mumax/3/data"
)

func dumpGnuplot(out io.Writer, f *data.Slice, m data.Meta) (err error) {
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
				_, err = fmt.Fprint(buf, x, " ", y, " ", z, "\t")
				for c := 0; c < ncomp; c++ {
					_, err = fmt.Fprint(buf, data[c][iz][iy][ix], " ")
				}
				_, err = fmt.Fprint(buf, "\n")
			}
			_, err = fmt.Fprint(buf, "\n")
		}
	}
	return
}
