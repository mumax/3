package data

type info struct {
	tag, unit string // Human-readable descriptors
	mesh      Mesh
}

// Human-readable tag to identify the data.
func (i *info) Tag() string { return i.tag }

// Physical unit of the data.
func (i *info) Unit() string { return i.unit }

// Mesh on which the data is defined.
func (i *info) Mesh() *Mesh { return &(i.mesh) }
