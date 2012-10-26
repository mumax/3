package gpu

import (
	"nimble-cube/core"
)

type RChan interface {
	Mesh() *core.Mesh
	NComp()int
	Size() [3]int
	Unit()string
NBlocks() int 
BlockLen() int 
}
