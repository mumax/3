package engine

import "code.google.com/p/mx3/data"

// holds quantity info for output etc.
type info struct {
	nComp      int        // number of components (scalar, vector, ...)
	name, unit string     // metadata for dump file
	mesh       *data.Mesh // nil means use global mesh
}

func (i *info) Mesh() *data.Mesh {
	if i.mesh == nil {
		return &globalmesh
	} else {
		return i.mesh
	}
}

func (i *info) NComp() int   { return i.nComp }
func (i *info) Name() string { return i.name }
func (i *info) Unit() string { return i.unit }
