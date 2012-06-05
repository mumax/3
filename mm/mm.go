package mm

import (
	"fmt"
	"log"
	. "nimble-cube/nc"
)

var (
	size [3]int // 3D geom size
	N    int    // product of size
	warp int    // buffer size for Range()

//	tChan      ScalarChan // Distributes time. Close means teardown listeners.
//	mChan      VecChan
//	alphaChan  Chan
//	hChan      VecChan
//	torqueChan VecChan
)

func Main() {

	initSize()

	torqueBox := new(TorqueBox)
	hBox := new(MeanFieldBox)
	solver := new(EulerBox)
	solver.dt = 0.01

	alpha := MakeFanIn()

	Connect3(&(hBox.m), &(solver.m))
	Connect3(&(torqueBox.m), &(solver.m))
	Connect3(&(torqueBox.h), &(hBox.h))
	Connect(&(torqueBox.alpha), &alpha)
	Connect3(&(solver.torque), &(torqueBox.t))

	//	Probe3(&(solver.m), "m")
	//	Probe3(&(hBox.h), "h")
	//	Probe3(&(torqueBox.t), "t")

	go torqueBox.Run()
	go hBox.Run()

	m0 := [3][]float32{make([]float32, N), make([]float32, N), make([]float32, N)}
	Memset3(m0, Vector{0.1, 0.99, 0})

	for i := 0; i < 1000; i++ {
		solver.Run(m0, 10)
		fmt.Println(m0[X][0], m0[Y][0], m0[Z][0])
	}

}

func DefaultBufSize() int { return N / warp }

func Connect3(dst *FanOut3, src *FanIn3) {
	buf := DefaultBufSize()
	*dst = src.FanOut(buf)
}

func initSize() {
	N0, N1, N2 := 1, 4, 8
	size := [3]int{N0, N1, N2}
	N = N0 * N1 * N2

	log.Println("size:", size)
	N := N0 * N1 * N2
	log.Println("N:", N)

	// Find some nice warp size
	warp = 8
	for N%warp != 0 {
		warp--
	}
	log.Println("warp:", warp)

}
