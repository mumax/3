package nimble

import (
	"code.google.com/p/nimble-cube/core"
	"fmt"
)

// Mesh stores info of a finite-difference mesh.
type Mesh struct {
	gridSize  [3]int
	cellSize  [3]float64
	pbc       [3]int
	blockSize [3]int
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
			core.Panic("mesh: need 0 or 3 PBC arguments, got:", pbc)
		}
	}
	size := [3]int{N0, N1, N2}
	return &Mesh{size, [3]float64{cellx, celly, cellz}, pbc3, blockSize(size)}
}

// Returns N0, N1, N2, as passed to constructor.
func (m *Mesh) Size() [3]int {
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

func (m *Mesh) WorldSize() [3]float64 {
	return [3]float64{float64(m.gridSize[0]) * m.cellSize[0], float64(m.gridSize[1]) * m.cellSize[1], float64(m.gridSize[2]) * m.cellSize[2]}
}

// Size of blocks in which the data on this mesh is divided.
// The data will have to buffer at least one block.
func (m *Mesh) BlockSize() [3]int {
	return m.blockSize
}

// Length of blocks in which the data on this mesh is divided.
// Product of the 3 dimensions returned by BlockSize.
func (m *Mesh) BlockLen() int {
	return prod(m.BlockSize())
}

func (m *Mesh) String() string {
	s := m.gridSize
	N := prod(s)
	c := m.cellSize
	pbc := ""
	if m.pbc != [3]int{0, 0, 0} {
		pbc = fmt.Sprintf("PBC: [%v x %v x %v],", m.pbc[0], m.pbc[1], m.pbc[2])
	}
	return fmt.Sprintf("%v cells: [%v x %v x %v ] x [%vm x %vm x %vm], %v blocksize: [%v x %v x %v]", N, s[0], s[1], s[2], c[0], c[1], c[2], pbc, m.blockSize[0], m.blockSize[1], m.blockSize[2])
}
