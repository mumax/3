package nimble

import ()

type info struct {
	tag, unit string // Human-readable descriptors
	mesh      Mesh
	MemType
}

func newInfo(m Mesh, mem MemType, tag, unit string) *info {
	return &info{tag, unit, m, mem}
}

func (i *info) Tag() string  { return i.tag }
func (i *info) Unit() string { return i.unit }
func (i *info) Mesh() *Mesh  { return &(i.mesh) }
