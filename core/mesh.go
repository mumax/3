package core

import "fmt"

// Mesh stores info of a finite-difference mesh.
type Mesh struct {
	gridSize [3]int
	cellSize [3]float64
	pbc      [3]int
}

// Retruns a new mesh with N0 x N1 x N2 cells of size cellx x celly x cellz.
// Optional periodic boundary conditions (pbc): number of repetitions
// in X, Y, Z direction. 0,0,0 means no periodicity.
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

// Returns N0, N1, N2, as passed to constructor.
func (m *Mesh) GridSize() [3]int {
	return m.gridSize
}

// Returns cellx, celly, cellz, as passed to constructor.
func (m *Mesh) CellSize() [3]float64 {
	return m.cellSize
}

// Returns pbc, as passed to constructor.
func (m *Mesh) PBC() [3]int {
	return m.pbc
}

// Total number of cells, not taking into account PBCs.
// 	N0 * N1 * N2
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

func (m *Mesh) String() string {
	s := m.gridSize
	N := Prod(s)
	c := m.cellSize
	pbc := m.pbc
	return fmt.Sprintf("%v cells: [%v x %v x %v ] x [%vm x %vm x %vm], PBC: [%v x %v x %v]", N, s[0], s[1], s[2], c[0], c[1], c[2], pbc[0], pbc[1], pbc[2])
}
