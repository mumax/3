package main

import (
	"log"
	"math"
	"strconv"
	"strings"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func resize(f *data.Slice, arg string) {
	s := parseSize(arg)
	resized := data.Resample(f, s)
	*f = *resized
}

func parseSize(arg string) (size [3]int) {
	words := strings.Split(arg, "x")
	if len(words) != 3 {
		log.Fatal("resize: need N0xN1xN2 argument")
	}
	for i, w := range words {
		v, err := strconv.Atoi(w)
		util.FatalErr(err)
		size[i] = v
	}
	return
}

func blurX(f *data.Slice, cells float64) {
	//TODO free memory again?

	Nx, Ny, Nz := f.Size()[0], f.Size()[1], f.Size()[2]

	//kernel
	kernel := make([][]float64, 2)
	for i := range kernel {
		kernel[i] = make([]float64, 1+int(math.Ceil(cells*6)))
	}
	prefactor := 1. / math.Sqrt(2*math.Pi*cells*cells)
	for i := range kernel[0] {
		position := float64(i) - float64(len(kernel[0])-1.)/2.
		kernel[0][i] = float64(position)
		kernel[1][i] = float64(prefactor * math.Exp(-1.*math.Pow(position, 2.)/(2*cells*cells)))
	}

	b := f.HostCopy()
	//empty f
	for i := 0; i < Nx; i++ {
		for j := 0; j < Ny; j++ {
			for k := 0; k < Nz; k++ {
				f.Set(0, i, j, k, 0.)
				f.Set(1, i, j, k, 0.)
				f.Set(2, i, j, k, 0.)
			}
		}
	}

	//convolution
	for i := 0; i < Nx; i++ {
		for j := 0; j < Ny; j++ {
			for k := 0; k < Nz; k++ {
				for l := range kernel[0] {
					if (i+int(kernel[0][l])) >= 0 && (i+int(kernel[0][l])) < Nx {
						f.Set(0, i, j, k, f.Get(0, i, j, k)+kernel[1][l]*b.Get(0, i+int(kernel[0][l]), j, k))
						f.Set(1, i, j, k, f.Get(1, i, j, k)+kernel[1][l]*b.Get(1, i+int(kernel[0][l]), j, k))
						f.Set(2, i, j, k, f.Get(2, i, j, k)+kernel[1][l]*b.Get(2, i+int(kernel[0][l]), j, k))
					}
				}
			}
		}
	}
	//normalize is necessary at the edges
	//	normalize(f, 1)
}

func blurY(f *data.Slice, cells float64) {
	//TODO free memory again?

	Nx, Ny, Nz := f.Size()[0], f.Size()[1], f.Size()[2]

	//kernel
	kernel := make([][]float64, 2)
	for i := range kernel {
		kernel[i] = make([]float64, 1+int(math.Ceil(cells*6)))
	}
	prefactor := 1. / math.Sqrt(2*math.Pi*cells*cells)
	for i := range kernel[0] {
		position := float64(i) - float64(len(kernel[0])-1.)/2.
		kernel[0][i] = float64(position)
		kernel[1][i] = float64(prefactor * math.Exp(-1.*math.Pow(position, 2.)/(2*cells*cells)))
	}

	b := f.HostCopy()
	//empty f
	for i := 0; i < Nx; i++ {
		for j := 0; j < Ny; j++ {
			for k := 0; k < Nz; k++ {
				f.Set(0, i, j, k, 0.)
				f.Set(1, i, j, k, 0.)
				f.Set(2, i, j, k, 0.)
			}
		}
	}

	//convolution
	for i := 0; i < Nx; i++ {
		for j := 0; j < Ny; j++ {
			for k := 0; k < Nz; k++ {
				for l := range kernel[0] {
					if (j+int(kernel[0][l])) >= 0 && (j+int(kernel[0][l])) < Ny {
						f.Set(0, i, j, k, f.Get(0, i, j, k)+kernel[1][l]*b.Get(0, i, int(kernel[0][l])+j, k))
						f.Set(1, i, j, k, f.Get(1, i, j, k)+kernel[1][l]*b.Get(1, i, int(kernel[0][l])+j, k))
						f.Set(2, i, j, k, f.Get(2, i, j, k)+kernel[1][l]*b.Get(2, i, int(kernel[0][l])+j, k))
					}
				}
			}
		}
	}
	//normalize is necessary at the edges
	//	normalize(f, 1)
}
