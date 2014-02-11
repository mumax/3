package engine

import (
	"github.com/mumax/3/data"
)

// Any space-dependent quantity
type Quantity interface {
	Slice() (q *data.Slice, recycle bool) // get quantity data (GPU or CPU), indicate need to recycle
	NComp() int
	Name() string
	Unit() string
	Mesh() *data.Mesh
	average() []float64
}
