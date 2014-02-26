// +build ignore

package main

import (
	"fmt"
	"github.com/barnex/fftw"
)

func main() {

	Ni := [2]int{3, 8}
	in := make([]float32, Ni[0]*Ni[1])
	In := fftw.ReshapeR2(in, Ni)

	No := [2]int{Ni[0], Ni[1]/2 + 1}
	out := make([]complex64, No[0]*No[1])
	Out := fftw.ReshapeC2(out, No)

	size := Ni[:]
	inembed := Ni[:]
	istride := 1
	idist := 0
	onembed := No[:]
	ostride := 1
	odist := 0
	p := fftw.PlanManyR2C(size[:], 1, in, inembed, istride, idist, out, onembed, ostride, odist, fftw.ESTIMATE)

	In[0][2] = 1
	for _, In := range In {
		fmt.Println(In)
	}

	fmt.Println()

	p.Execute()

	for _, Out := range Out {
		fmt.Println(Out)
	}
}
