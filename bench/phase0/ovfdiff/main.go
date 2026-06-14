package main

import (
	"fmt"
	"math"
	"os"

	"github.com/mumax/3/oommf"
)

func main() {
	a, _, err := oommf.ReadFile(os.Args[1])
	if err != nil {
		panic(err)
	}
	b, _, err := oommf.ReadFile(os.Args[2])
	if err != nil {
		panic(err)
	}

	sa, sb := a.Tensors(), b.Tensors()
	size := a.Size()
	nc := a.NComp()

	maxDiff := float32(0)
	maxAbs := float32(0)
	var maxAt [4]int
	ndiff := 0
	total := 0
	for c := 0; c < nc; c++ {
		for iz := 0; iz < size[2]; iz++ {
			for iy := 0; iy < size[1]; iy++ {
				for ix := 0; ix < size[0]; ix++ {
					va := sa[c][iz][iy][ix]
					vb := sb[c][iz][iy][ix]
					d := float32(math.Abs(float64(va - vb)))
					total++
					if d != 0 {
						ndiff++
						if d > maxDiff {
							maxDiff = d
							maxAt = [4]int{c, ix, iy, iz}
						}
					}
					if float32(math.Abs(float64(va))) > maxAbs {
						maxAbs = float32(math.Abs(float64(va)))
					}
				}
			}
		}
	}
	fmt.Printf("size=%v nComp=%d total=%d ndiff=%d maxDiff=%.9g maxAbs=%.9g maxAt(c,x,y,z)=%v\n",
		size, nc, total, ndiff, maxDiff, maxAbs, maxAt)
	if maxDiff != 0 {
		c, x, y, z := maxAt[0], maxAt[1], maxAt[2], maxAt[3]
		fmt.Printf("a[%d][%d][%d][%d]=%.9g  b=...=%.9g\n", c, z, y, x, sa[c][z][y][x], sb[c][z][y][x])
	}
}
