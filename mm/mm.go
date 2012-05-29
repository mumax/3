package mm

import (
	"log"
	. "nimble-cube/nc"
)

var (
	size   [3]int      // 3D geom size
	N      int         // product of size
	warp   int         // buffer size for Range()
	m0, m1 VectorBlock // reduced magnetization, at old and new t
	heff   VectorBlock // effective field
	gamma  float32     // gyromagnetic ratio
	alpha  float32     // damping coefficient
	torque VectorBlock // dm/dt
)

func Main() {

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

	m0 = MakeVectorBlock(size)
	heff = MakeVectorBlock(size)
	torque = MakeVectorBlock(size)

	m0[X].Memset(1)
	m0[Y].Memset(0)
	m0[Z].Memset(0)

	UpdateTorque()
	log.Println(torque)
}

func UpdateTorque() {
	N := torque.NVector()

	τx := torque[X].Contiguous()
	τy := torque[X].Contiguous()
	τz := torque[X].Contiguous()

	M := m0.Contiguous()
	//H := heff.Contiguous()

	for I := 0; I < N; I += warp {

		torque := Torque.Range(I, I+warp)
		m := M.Range(I, I+warp)
		//heff := H.Range(I, I+warp)

		for i := 0; i < warp; i++ {
			torque[i] = m[i]
		}
	}
}
