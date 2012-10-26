package gpu

import (
	"nimble-cube/core"
	//"github.com/barnex/cuda5/safe"
)

type Info interface {
	Mesh() *core.Mesh
	Size() [3]int
	Unit() string
	NBlocks() int
	BlockLen() int
}

type RChan interface {
	Info
	NComp() int
	Comp(int) RChan1
}
