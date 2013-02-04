package nimble

import ()

type info struct {
	tag, unit string // Human-readable descriptors
	mesh      Mesh
	MemType
}

func newInfo(tag, unit string, m Mesh, mem MemType) info {
	return info{tag, unit, m, mem}
}

func (i *info) Tag() string  { return i.tag }
func (i *info) Unit() string { return i.unit }
func (i *info) Mesh() *Mesh  { return &(i.mesh) }
