package data

import (
	"code.google.com/p/mx3/util"
	"fmt"
	"log"
)

// Mesh stores info of a finite-difference mesh.
type Mesh struct {
	gridSize [3]int
	cellSize [3]float64
	pbc      [3]int
	Unit     string // unit of cellSize, default: "m"
}

// Retruns a new mesh with N0 x N1 x N2 cells of size cellx x celly x cellz.
// Optional periodic boundary conditions (pbc): number of repetitions
// in X, Y, Z direction. 0,0,0 means no periodicity.
func NewMesh(N0, N1, N2 int, cellx, celly, cellz float64, pbc ...int) *Mesh {
	util.Argument(N0 > 0 && N1 > 0 && N2 > 0)
	util.Argument(cellx > 0 && celly > 0 && cellz > 0)
	var pbc3 [3]int
	if len(pbc) == 3 {
		copy(pbc3[:], pbc)
	} else {
		if len(pbc) != 0 {
			log.Panic("mesh: need 0 or 3 PBC arguments, got:", pbc)
		}
	}
	size := [3]int{N0, N1, N2}
	return &Mesh{size, [3]float64{cellx, celly, cellz}, pbc3, "m"}
}

// Returns N0, N1, N2, as passed to constructor.
func (m *Mesh) Size() [3]int {
	return m.gridSize
}

// Returns cellx, celly, cellz, as passed to constructor.
func (m *Mesh) CellSize() [3]float64 {
	return m.cellSize
}

// Returns pbc (periodic boundary conditions), as passed to constructor.
func (m *Mesh) PBC() [3]int {
	return m.pbc
}

// Total number of cells, not taking into account PBCs.
// 	N0 * N1 * N2
func (m *Mesh) NCell() int {
	return m.gridSize[0] * m.gridSize[1] * m.gridSize[2]
}

// WorldSize equals (grid)Size x CellSize.
func (m *Mesh) WorldSize() [3]float64 {
	return [3]float64{float64(m.gridSize[0]) * m.cellSize[0], float64(m.gridSize[1]) * m.cellSize[1], float64(m.gridSize[2]) * m.cellSize[2]}
}

// String representation in internal coordinates (ZYX)
//func (m *Mesh) InternString() string {
//	s := m.gridSize
//	c := m.cellSize
//	pbc := ""
//	if m.pbc != [3]int{0, 0, 0} {
//		pbc = fmt.Sprintf(", PBC: [%v x %v x %v],", m.pbc[0], m.pbc[1], m.pbc[2])
//	}
//	return fmt.Sprintf("[%v x %v x %v] x [%vm x %vm x %vm]%v", s[0], s[1], s[2], c[0], c[1], c[2], pbc)
//}

// String representation in user coordinates (XYZ)
func (m *Mesh) UserString() string {
	s := m.gridSize
	c := m.cellSize
	pbc := ""
	if m.pbc != [3]int{0, 0, 0} {
		pbc = fmt.Sprintf(", PBC: [%v x %v x %v],", m.pbc[2], m.pbc[1], m.pbc[0])
	}
	return fmt.Sprintf("[%v x %v x %v] x [%vm x %vm x %vm]%v", s[2], s[1], s[0], float32(c[2]), float32(c[1]), float32(c[0]), pbc)
}

// product of elements.
func prod(size [3]int) int {
	return size[0] * size[1] * size[2]
}
