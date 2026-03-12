package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func init() {
	DeclFunc("CustomQuantity", CustomQuantity, "Creates a custom Quantity from the user provided (scalar or vector) slice.")
}

func CustomQuantity(inSlice *data.Slice) Quantity {
	util.Assert(inSlice.NComp() == 1 || inSlice.NComp() == 3)
	size := Mesh().Size()
	sliceSize := inSlice.Size()
	util.Assert(size[X] == sliceSize[X] && size[Y] == sliceSize[Y] && size[Z] == sliceSize[Z])

	retQuant := &customQuantity{nil, size}
	if inSlice.NComp() == 1 {
		retQuant.customquant = cuda.NewSlice(SCALAR, size)
	} else {
		retQuant.customquant = cuda.NewSlice(VECTOR, size)
	}

	data.Copy(retQuant.customquant, inSlice)
	return retQuant
}

type customQuantity struct {
	customquant *data.Slice
	size        [3]int
}

func (q *customQuantity) NComp() int {
	return q.customquant.NComp()
}

func (q *customQuantity) EvalTo(dst *data.Slice) {
	util.Assert(dst.NComp() == q.customquant.NComp())
	data.Copy(dst, q.customquant)
}
