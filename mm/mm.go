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

	// 0) initialize size, warp, etc
	initSize()

	// 1) make and connect boxes
	torqueBox := new(TorqueBox)
	hBox := new(MeanFieldBox)
	solver := new(EulerBox)
	solver.dt = 0.01
	alphaBox := NewConstBox(0.1)

	Connect(&hBox, "m", &solver, "m")
	//Connect3(&(torqueBox.m), &(solver.m))
	//Connect3(&(torqueBox.h), &(hBox.h))
	//Connect(&(torqueBox.alpha), &(alphaBox.output))
	//Connect3(&(solver.torque), &(torqueBox.t))

	//	Probe3(&(solver.m), "m")
	//	Probe3(&(hBox.h), "h")
	//	Probe3(&(torqueBox.t), "t")

	// 3) run boxes, no more should be created from now
	go torqueBox.Run()
	go hBox.Run()
	go alphaBox.Run()

	m0 := [3][]float32{make([]float32, N), make([]float32, N), make([]float32, N)}
	Memset3(m0, Vector{0.1, 0.99, 0})

	// Solver box runs synchronous.
	// Could be async with return channel...
	for i := 0; i < 1000; i++ {
		solver.Run(m0, 10)
		fmt.Println(m0[X][0], m0[Y][0], m0[Z][0])
	}

	// 4) tear-down and wait for boxes to finish
	// needed to cleanly close down output boxes, e.g.
	// Use runtime.NumGoroutine() to assert all is well and panic(Bug()) otherwise
	// to enforce good implementation.
	// ...
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
