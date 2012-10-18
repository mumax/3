package gpu

import (
	"fmt"
	"github.com/barnex/cuda5/safe"
	"nimble-cube/core"
)

type Quant struct {
	tag, unit string
	*core.Mesh
	list    []safe.Float32s
	nBlocks int
}

func NewQuant(tag string, nComp int, m *core.Mesh, unit string, nBlocks ...int) *Quant {
	q := &Quant{tag: core.Unique(tag), unit: unit, Mesh: m}

	blocklen := core.BlockLen(q.Size())
	maxBlocks := q.NCell() / blocklen
	if len(nBlocks) > 1 {
		core.Fatal(fmt.Errorf("newquant: nblocks... should be ≤ 1 parameter"))
	}
	blocks := maxBlocks
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

func (q*Quant)NComp()int{
	return len(q.list)
}

func(q*Quant)String()string{
	return fmt.Sprint(q.tag, ": ", q.NComp(), "x", q.Size(), ", ", q.nBlocks, " blocks")	
}
