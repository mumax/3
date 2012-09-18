package core

type Mesh struct {
	gridSize [3]int
	cellSize [3]float64
	pbc      [3]int
}

func NewMesh(N0, N1, N2 int, cellx, celly, cellz float64, pbc ...int) *Mesh {
	var pbc3 [3]int
	if len(pbc) == 3 {
		copy(pbc3[:], pbc)
	} else {
		if len(pbc) != 0 {
			Panic("mesh: need 0 or 3 PBC arguments, got:", pbc)
		}
	}
	return &Mesh{[3]int{N0, N1, N2}, [3]float64{cellx, celly, cellz}, pbc3}
}

func (m *Mesh) GridSize() [3]int {
	return m.gridSize
}

func (m *Mesh) CellSize() [3]float64 {
	return m.cellSize
}

func (m *Mesh) PBC() [3]int {
	return m.pbc
}

// Total number of cells, not taking into account PBCs.
func (m *Mesh) NCell() int {
	return m.gridSize[0] * m.gridSize[1] * m.gridSize[2]
}

// Returns the mesh size after zero-padding.
// The zero padded size in any direction is twice
// the original size unless the original size was
// 1 or unless there are PBCs in that direction.
func (m *Mesh) ZeroPadded() *Mesh {
	padded := padSize(m.gridSize, m.pbc)
	return &Mesh{padded, m.cellSize, m.pbc}
}

// Returns the size after zero-padding,
// taking into account periodic boundary conditions.
func padSize(size, periodic [3]int) [3]int {
	for i := range size {
		if periodic[i] == 0 && size[i] > 1 {
			size[i] *= 2
		}
	}
	return size
}
