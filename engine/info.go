package engine

import "code.google.com/p/mx3/data"

// holds quantity info for output etc.
type info struct {
	doc
	mesh *data.Mesh // nil means use global mesh
}

func mkInfo(nComp int, name, unit string, m *data.Mesh) info {
	return info{doc{nComp, name, unit}, m}
}

type doc struct {
	nComp      int    // number of components (scalar, vector, ...)
	name, unit string // metadata for dump file
}

func (i *info) Mesh() *data.Mesh { return i.mesh }
func (i *doc) NComp() int        { return i.nComp }
func (i *doc) Name() string      { return i.name }
func (i *doc) Unit() string      { return i.unit }
