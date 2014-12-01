package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/barnex/matrix"
	"github.com/mumax/3/data"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/oommf"
)

func mainSpatial() {

	Nt := flag.NArg()    // time points
	Nf := 2 * (Nt/2 + 1) // frequency points

	comp := 0 // todo

	data1, meta1 := oommf.MustReadFile(flag.Args()[0])
	_, metaL := oommf.MustReadFile(flag.Args()[Nt-1])

	size := data1.Size()
	t0 := float32(meta1.Time)
	t1 := float32(metaL.Time)

	Nx := size[0]
	Ny := size[1]
	Nz := size[2]

	dataList := make([]float32, Nf*Nx*Ny*Nz)
	dataLists := matrix.ReshapeR2(dataList, [2]int{Nf, Nz * Ny * Nx})
	//dataArr := matrix.ReshapeR4(dataList, [4]int{Nt, Nz, Ny, Nx})

	deltaT := t1 - t0

	time0 := t0                  // start time, not neccesarily 0
	si := 0                      // source index
	for di := 0; di < Nt; di++ { // dst index
		want := time0 + float32(di)*deltaT/float32(Nt) // wanted time
		for si < Nt-1 && !(time(si) <= want && time(si+1) > want && time(si) != time(si+1)) {
			si++
		}

		x := (want - time(si)) / (time(si+1) - time(si))
		if x < 0 || x > 1 {
			panic(fmt.Sprint("x=", x))
		}
		interp3D(dataLists[di], 1-x, file(si).Host()[comp], x, file(si + 1).Host()[comp])
	}

	output3D(dataLists, size, "interp", deltaT)
}

func output3D(d [][]float32, size [3]int, prefix string, deltaT float32) {
	for i, d := range d {
		fname := fmt.Sprintf("%s%06d.ovf", prefix, i)
		slice := data.SliceFromArray([][]float32{d}, size)
		meta := data.Meta{}
		f := httpfs.MustCreate(fname)
		oommf.WriteOVF2(f, slice, meta, "binary")
		f.Close()
	}
}

func interp3D(dst []float32, w1 float32, src1 []float32, w2 float32, src2 []float32) {
	for i := range dst {
		dst[i] = w1*src1[i] + w2*src2[i]
	}
}

var (
	prevData    *data.Slice
	prevMeta    data.Meta
	cachedData  *data.Slice
	cachedMeta  data.Meta
	cachedIndex int = -1
)

func file(i int) *data.Slice {
	d, _ := loadFile(i)
	return d
}

func time(i int) float32 {
	_, m := loadFile(i)
	return float32(m.Time)
}

func loadFile(i int) (*data.Slice, data.Meta) {
	if i > cachedIndex+1 || i < cachedIndex-1 {
		panic(fmt.Sprintf("index out-of-order: %v (previous: %v)", i, cachedIndex))
	}
	if i == cachedIndex-1 {
		return prevData, prevMeta
	}
	if i == cachedIndex {
		return cachedData, cachedMeta
	}
	if i == cachedIndex+1 {
		prevData, prevMeta = cachedData, cachedMeta
		fname := flag.Args()[i]
		log.Println("loading", fname)
		cachedData, cachedMeta = oommf.MustReadFile(fname) // TODO: preprocess here
		cachedIndex = i
		return cachedData, cachedMeta
	}
	panic("bug")
}
