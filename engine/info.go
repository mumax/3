package engine

func Info(nComp int, name, unit string) info {
	return info{nComp: nComp, name: name, unit: unit}
}

type info struct {
	nComp      int // number of components (scalar, vector, ...)
	name, unit string
}

func (i *info) Name() string { return i.name }
func (i *info) Unit() string { return i.unit }
func (i *info) NComp() int   { return i.nComp }
