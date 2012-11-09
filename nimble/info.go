package nimble

import ()

type Info struct {
	tag, unit string // Human-readable descriptors
	*Mesh
}

func NewInfo(tag, unit string, m *Mesh) *Info {
	return &Info{tag, unit, m}
}

func (i *Info) Tag() string  { return i.tag }
func (i *Info) Unit() string { return i.unit }
