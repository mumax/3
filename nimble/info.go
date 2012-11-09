package nimble

import (
)

type Info struct {
	tag, unit string // Human-readable descriptors
	*Mesh
}

func NewInfo(tag, unit string, m *Mesh) *Info {
	return &Info{tag, unit, m}
}

func (i *Info) Tag() string   { return i.tag }
func (i *Info) Unit() string  { return i.unit }

//func parseNBlocks(size [3]int, nBlocks ...int) (blocks, blocklen int) {
//	blocklen = BlockLen(size)
//	maxBlocks := Prod(size) / blocklen
//	if len(nBlocks) > 1 {
//		core.Fatalf("newquant: nblocks... should be ≤ 1 parameter")
//	}
//	blocks = maxBlocks // TODO: both maxblocks or 1 are good choices here
//	if len(nBlocks) > 0 {
//		blocks = nBlocks[0]
//	}
//	if blocks > maxBlocks { // must not use more blocks than possible.
//		blocks = maxBlocks
//	}
//	return
//}
