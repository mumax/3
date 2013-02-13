package mx

type info struct {
	tag, unit string // Human-readable descriptors
	mesh      Mesh
}

func (i *info) Tag() string  { return i.tag }
func (i *info) Unit() string { return i.unit }
func (i *info) Mesh() *Mesh  { return &(i.mesh) }
