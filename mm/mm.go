package mm

import (
	"log"
	//"nimble-cube/nc"
)

var (
	size [3]int // 3D geom size
	N    int    // product of size
	warp int    // buffer size for Range()

	//m0, m1 VectorBlock // reduced magnetization, at old and new t
	//heff   VectorBlock // effective field
	//gamma  float32     // gyromagnetic ratio
	//alpha  float32     // damping coefficient
	//torque VectorBlock // dm/dt
)

func Main() {

	initSize()

	m := MakeDoubleBuffer(N)

	pipe := MakePipe()
	m.SendPipe = pipe.SendPipe()
	m.RecvPipe = pipe.RecvPipe()

	m.Cycle()
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
