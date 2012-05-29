package mm

import (
	. "nimble-cube/nc"
)

var (
	size   [3]int      // 3D geom size
	warp   int         // buffer size for Range()
	m      VectorBlock // reduced magnetization
	heff   VectorBlock // effective field
	gamma  float32
	alpha  float32
	torque VectorBlock
)

func Main() {

	N0, N1, N2 := 1, 4, 8
	size := [3]int{N0, N1, N2}

	Println("size:", size)
	N := N0 * N1 * N2
	Println("N:", N)

	// Find some nice warp size
	warp = 8
	for N%warp != 0 {
		warp--
	}
	Println("warp:", warp)

	m = MakeVectorBlock(size)
	heff = MakeVectorBlock(size)
	torque = MakeVectorBlock(size)

	m[X].Memset(1)
	m[Y].Memset(0)
	m[Z].Memset(0)

	UpdateTorque()
	Println(torque)
}

func UpdateTorque() {
	N := torque.NVector()
	Torque := torque.Contiguous()
	M := m.Contiguous()
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
