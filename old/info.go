package nimble

import ()

type info struct {
	tag, unit string // Human-readable descriptors
	mesh      Mesh
}

func newInfo(m Mesh, tag, unit string) *info {
	return &info{tag, unit, m}
}

func (i *info) Tag() string  { return i.tag }
func (i *info) Unit() string { return i.unit }
func (i *info) Mesh() *Mesh  { return &(i.mesh) }
