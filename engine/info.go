package engine

import "github.com/mumax/3/data"

type descr struct {
	name, unit string
}

type doc struct {
	nComp int // number of components (scalar, vector, ...)
	descr
}

func Doc(nComp int, name, unit string) doc {
	return doc{nComp, descr{name, unit}}
}

type info struct {
	doc
	mesh *data.Mesh // nil means use global mesh
}

func Info(nComp int, name, unit string, m *data.Mesh) info {
	return info{doc{nComp, descr{name, unit}}, m}
}

func (d *descr) Name() string    { return d.name }
func (d *descr) Unit() string    { return d.unit }
func (i *doc) NComp() int        { return i.nComp }
func (i *info) Mesh() *data.Mesh { return i.mesh }
