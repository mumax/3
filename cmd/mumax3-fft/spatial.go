package main

import (
	"flag"
	"fmt"
	"log"
	"math"

	"github.com/barnex/fftw"
	"github.com/barnex/matrix"
	"github.com/mumax/3/data"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/oommf"
)

func mainSpatial() {

	// time points
	Nt := flag.NArg()
	if Nt < 2 {
		log.Fatal("need at least 2 inputs")
	}

	go loadloop()

	// select one component
	comp := 0 // todo

	// get size, time span from first and last file
	data1, meta1 := oommf.MustReadFile(flag.Args()[0])
	_, metaL := oommf.MustReadFile(flag.Args()[Nt-1])

	t0 := float32(meta1.Time)
	t1 := float32(metaL.Time)
	deltaT := t1 - t0
	deltaF := 1 / deltaT // frequency resolution

	size := data1.Size()
	Nx := size[0]
	Ny := size[1]
	Nz := size[2]

	// allocate buffer for everything
	dataList := make([]complex64, Nt*Nx*Ny*Nz)
	dataLists := matrix.ReshapeC2(dataList, [2]int{Nt, Nz * Ny * Nx})

	// interpolate non-equidistant time points
	// make complex in the meanwhile
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

	log.Println("FFT")
	fftMany(dataList, Nt, Nx*Ny*Nz)

	spectrum := magSpectrum(dataLists)
	freqs := make([]float32, len(spectrum))
	for i := range freqs {
		freqs[i] = float32(i) * deltaF
	}
	header := []string{"f (Hz)", "Mag ()"}
	f := httpfs.MustCreate("spectrum.txt")
	writeTable(f, header, [][]float32{freqs, spectrum})
	f.Close()

	log.Println("normalize")
	normalize(dataLists)

	log.Println("output")
	for _, o := range outputs {
		if *o.Enabled {
			output3D(dataLists, o.Filter, size, "fft_"+o.Name, deltaF)
		}
	}
}

func magSpectrum(dataLists [][]complex64) []float32 {
	Nf := len(dataLists) / 2
	spec := make([]float32, Nf)
	for f := range spec {
		sum := 0.
		for _, v := range dataLists[f] {
			sum += float64(real(v)*real(v) + imag(v)*imag(v))
		}
		sum = math.Sqrt(sum)
		spec[f] = float32(sum)
	}
	return spec
}

// normalize all but DC
func normalize(dataList [][]complex64) {
	var max float32
	for j := 1; j < len(dataList); j++ { // skip DC
		for i := range dataList[j] {
			v := dataList[j][i]
			norm2 := real(v)*real(v) + imag(v)*imag(v)
			if norm2 > max {
				max = norm2
			}
		}
	}
	norm := complex(float32(1/math.Sqrt(float64(max))), 0)

	for _, dataList := range dataList {
		for i := range dataList {
			dataList[i] *= norm
		}
	}
}

func fftMany(dataList []complex64, Nt, Nc int) {
	howmany := Nc
	n := []int{Nt}
	in := dataList
	out := dataList
	istride := Nc
	idist := 1
	inembed := n
	ostride := istride
	odist := idist
	onembed := inembed
	plan := fftw.PlanManyC2C(n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, fftw.FORWARD, fftw.ESTIMATE)
	plan.Execute()
	//plan.Destroy()
}

func output3D(D [][]complex64, reduce func(complex64) float32, size [3]int, prefix string, deltaF float32) {
	const NCOMP = 1
	for i := 0; i < len(D)/2; i++ {
		d := D[i]
		MHz := int((float32(i) * deltaF) / 1e6)
		fname := fmt.Sprintf("%sf%06dMHz.ovf", prefix, MHz)
		slice := data.NewSlice(NCOMP, size)
		doReduce(slice.Host()[0], d, reduce)
		meta := data.Meta{}
		log.Println(fname)
		f := httpfs.MustCreate(fname)
		oommf.WriteOVF2(f, slice, meta, "binary")
		f.Close()
	}
}

func doReduce(dst []float32, src []complex64, f func(complex64) float32) {
	for i := range dst {
		dst[i] = f(src[i])
	}
}

func interp3D(dst []complex64, w1 float32, src1 []float32, w2 float32, src2 []float32) {
	for i := range dst {
		dst[i] = complex(w1*src1[i]+w2*src2[i], 0)
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
		cachedIndex++
		inp := <-inpipe
		cachedData = inp.D
		cachedMeta = inp.M
		return cachedData, cachedMeta
	}
	panic("bug")
}

var inpipe = make(chan inp, 2)

func loadloop() {
	for _, fname := range flag.Args() {
		log.Println("loading", fname)
		cachedData, cachedMeta = oommf.MustReadFile(fname) // TODO: preprocess here
		inpipe <- inp{cachedData, cachedMeta}
	}
	close(inpipe)
}

type inp struct {
	D *data.Slice
	M data.Meta
}
