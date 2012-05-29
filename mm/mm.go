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

	gamma = 1
	log.Println("gamma:", gamma)

	// Find some nice warp size
	warp = 8
	for N%warp != 0 {
		warp--
	}
	log.Println("warp:", warp)

	m0 = MakeVectorBlock(size)
	heff = MakeVectorBlock(size)
	torque = MakeVectorBlock(size)

	Vecset(m0.Contiguous(), Vector{1, 0, 0})
	Vecset(heff.Contiguous(), Vector{0, 1, 0})

	UpdateTorque()
	log.Println(torque)
}

func UpdateTorque() {

	for I := 0; I < N; I += warp {

		τ_ := torque.Range(I, I+warp)
		m_ := m0.Range(I, I+warp)
		h_ := heff.Range(I, I+warp)

		for i := 0; i < warp; i++ {
			m := Vector{m_[X][i], m_[Y][i], m_[Z][i]}
			h := Vector{h_[X][i], h_[Y][i], h_[Z][i]}

			τ := m.Cross(h)

			τ_[X][i] = gamma * τ[X]
			τ_[Y][i] = gamma * τ[Y]
			τ_[Z][i] = gamma * τ[Z]
		}
	}
}
