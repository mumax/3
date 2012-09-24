package mag

import (
	"nimble-cube/core"
)

type Exchange6 struct {
	m   core.RChan3
	hex core.Chan3
	*core.Mesh
	aex_reduced float64
}

func (e *Exchange6) Run() {
	// TODO: properly split in blocks
	for {
		m := core.Reshape3(e.m.ReadNext(e.NCell()), e.GridSize())
		hex := core.Reshape3(e.hex.WriteNext(e.NCell()), e.GridSize())
		exchange6(m, hex, e.CellSize(), e.aex_reduced)
		e.hex.WriteDone()
		e.m.ReadDone()
	}
}

// Naive implementation of 6-neighbor exchange field.
// Aex in Tm² (exchange stiffness divided by Msat0).
// Hex in Tesla.
func exchange6(m [3][][][]float32, Hex [3][][][]float32, cellsize [3]float64, aex_reduced float64) {
	var (
		facI = float32(aex_reduced / (cellsize[0] * cellsize[0]))
		facJ = float32(aex_reduced / (cellsize[1] * cellsize[1]))
		facK = float32(aex_reduced / (cellsize[2] * cellsize[2]))
	)
	N0, N1, N2 := len(m[0]), len(m[0][0]), len(m[0][0][0])

	for i := range m[0] {
		// could hack in here (for 3D)...
		for j := range m[0][i] {
			// or here (for 2D) and synchronize
			for k := range m[0][i][j] {
				m0 := Vector{m[X][i][j][k],
					m[Y][i][j][k],
					m[Z][i][j][k]}
				var hex, m_neigh Vector

				if i > 0 {
					m_neigh = Vector{m[X][i-1][j][k],
						m[Y][i-1][j][k],
						m[Z][i-1][j][k]}
					hex = hex.Add((m_neigh.Sub(m0)).Scaled(facI))
				}

				if i < N0-1 {
					m_neigh = Vector{m[X][i+1][j][k],
						m[Y][i+1][j][k],
						m[Z][i+1][j][k]}
					hex = hex.Add((m_neigh.Sub(m0)).Scaled(facI))
				}

				if j > 0 {
					m_neigh = Vector{m[X][i][j-1][k],
						m[Y][i][j-1][k],
						m[Z][i][j-1][k]}
					hex = hex.Add((m_neigh.Sub(m0)).Scaled(facJ))
				}

				if j < N1-1 {
					m_neigh = Vector{m[X][i][j+1][k],
						m[Y][i][j+1][k],
						m[Z][i][j+1][k]}
					hex = hex.Add((m_neigh.Sub(m0)).Scaled(facJ))
				}

				if k > 0 {
					m_neigh = Vector{m[X][i][j][k-1],
						m[Y][i][j][k-1],
						m[Z][i][j][k-1]}
					hex = hex.Add((m_neigh.Sub(m0)).Scaled(facK))
				}

				if k < N2-1 {
					m_neigh = Vector{m[X][i][j][k+1],
						m[Y][i][j][k+1],
						m[Z][i][j][k+1]}
					hex = hex.Add((m_neigh.Sub(m0)).Scaled(facK))
				}

				Hex[X][i][j][k] = hex[X]
				Hex[Y][i][j][k] = hex[Y]
				Hex[Z][i][j][k] = hex[Z]
			}
		}
	}
}
