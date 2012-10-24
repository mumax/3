package core

type Info struct {
	tag, unit string // Human-readable descriptors
	*Mesh
	nBlocks, blockLen int
}

func NewInfo(tag, unit string, m *Mesh, nBlocks, blocklen int)*Info{
	return&Info{tag, unit, m, nBlocks, blocklen}
}

func(i*Info) Tag()string{return i.tag}
func(i*Info) Unit()string{return i.unit}
func(i*Info) NBlocks()int{return i.nBlocks}
func(i*Info) BlockLen()int{return i.blockLen}

