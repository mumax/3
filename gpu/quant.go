package gpu

import (
	"fmt"
	"github.com/barnex/cuda5/safe"
	"nimble-cube/core"
)

// Quant stores a physical on GPU.
type Quant struct {
	tag, unit string
	*core.Mesh
	list    []safe.Float32s
	nBlocks int
}

// NewQuant creates a new physical quantity.
// Optional nBlocks: number of storage blocks to use. 
// Typically 1 for local quantities (e.g.: zeeman field).
// Typically 2 or 3 for stencil inputs (e.g.: input for exchange interaction).
// 0 (default) means fully store the quantity, typical for global quantities like demag input. 
// 0 is always safe wrt. deadlock but may use more memory than strictly needed.
func NewQuant(tag string, nComp int, m *core.Mesh, unit string, nBlocks ...int) *Quant {
	q := &Quant{tag: core.Unique(tag), unit: unit, Mesh: m}

	blocklen := core.BlockLen(q.Size())
	maxBlocks := q.NCell() / blocklen
	if len(nBlocks) > 1 {
		core.Fatal(fmt.Errorf("newquant: nblocks... should be ≤ 1 parameter"))
	}
	blocks := maxBlocks // TODO: both maxblocks or 1 are good choices here
	if len(nBlocks) > 0 {
		blocks = nBlocks[0]
	}
	if blocks > maxBlocks {
		blocks = maxBlocks
	}

	q.nBlocks = blocks
	q.list = make([]safe.Float32s, nComp)
	for i := range q.list {
		q.list[i] = safe.MakeFloat32s(blocks * blocklen)
	}

	return q
}

// NComp returns the number of components
// (1: scalar, 3: vector, ...)
func (q *Quant) NComp() int {
	return len(q.list)
}

// Unit returns the quantity's physical unit.
func(q*Quant)Unit()string{
	return q.unit
}

func (q *Quant) String() string {
	unit := q.unit
	if unit != "" {
		unit = " [" + unit + "]"
	}
	return fmt.Sprint(q.tag, unit, ": ", q.NComp(), "x", q.Size(), ", ", q.nBlocks, " blocks")
}
