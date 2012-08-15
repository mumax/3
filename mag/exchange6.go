package mag

// Naive implementation of 6-neighbor exchange field.
func Exchange6(m [3][][][]float32, Hex [3][][][]float32) {
	var (
		facI float32 = 1
		facJ float32 = 1
		facK float32 = 1
	)
	N0, N1, N2 := len(m[0]), len(m[0][0]), len(m[0][0][0])

	for i := range m[0] {
		for j := range m[0][i] {
			for k := range m[0][i][j] {
				m0 := Vector{m[X][i][j][k],
					m[Y][i][j][k],
					m[Z][i][j][k]}
				var hex, m_neigh Vector

				if i > 0 {
					m_neigh = Vector{m[X][i-1][j][k],
						m[Y][i-1][j][k],
						m[Z][i-1][j][k]}
					hex = hex.Add((m0.Sub(m_neigh)).Scaled(facI))
				}

				if i < N0-1 {
					m_neigh = Vector{m[X][i+1][j][k],
						m[Y][i+1][j][k],
						m[Z][i+1][j][k]}
					hex = hex.Add((m0.Sub(m_neigh).Scaled(facI)))
				}

				if j > 0 {
					m_neigh = Vector{m[X][i][j-1][k],
						m[Y][i][j-1][k],
						m[Z][i][j-1][k]}
					hex = hex.Add((m0.Sub(m_neigh)).Scaled(facJ))
				}

				if j < N1-1 {
					m_neigh = Vector{m[X][i][j+1][k],
						m[Y][i][j+1][k],
						m[Z][i][j+1][k]}
					hex = hex.Add((m0.Sub(m_neigh).Scaled(facJ)))
				}

				if k > 0 {
					m_neigh = Vector{m[X][i][j][k-1],
						m[Y][i][j][k-1],
						m[Z][i][j][k-1]}
					hex = hex.Add((m0.Sub(m_neigh)).Scaled(facK))
				}

				if k < N2-1 {
					m_neigh = Vector{m[X][i][j][k+1],
						m[Y][i][j][k+1],
						m[Z][i][j][k+1]}
					hex = hex.Add((m0.Sub(m_neigh).Scaled(facK)))
				}

				Hex[X][i][j][k] = hex[X]
				Hex[Y][i][j][k] = hex[Y]
				Hex[Z][i][j][k] = hex[Z]
			}
		}
	}
}
