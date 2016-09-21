package engine

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

func init() {
	DeclFunc("CropToRegion", CropToRegion, "Restrict a quantity to a region, return zero elsewhere.")
}

// represents a new quantity equal to q in the given region, 0 outside.
type oneReg struct {
	parent Q
	region int
}

func InRegion(q Q, r int) Q {
	return &oneReg{q, r}
}

type sOneReg struct{ oneReg }

func (q *sOneReg) Average() float64 { return q.average()[0] }

type vOneReg struct{ oneReg }

func (q *vOneReg) Average() data.Vector { return unslice(q.average()) }

func CropToRegion(q Q, region int) Q {
	return &oneReg{q, region}
}

func (q *oneReg) NComp() int   { return q.parent.NComp() }
func (q *oneReg) Name() string { return fmt.Sprint(NameOf(q.parent), ".region", q.region) }

func (q *oneReg) EvalTo(dst *data.Slice) {
	q.parent.EvalTo(dst)
	cuda.RegionSelect(dst, dst, regions.Gpu(), byte(q.region))
}

func (q *oneReg) average() []float64 {
	slice := ValueOf(q)
	defer cuda.Recycle(slice)
	avg := sAverageUniverse(slice)
	sDiv(avg, regions.volume(q.region))
	return avg
}

func (q *oneReg) Average() []float64 { return q.average() }

// slice division
func sDiv(v []float64, x float64) {
	for i := range v {
		v[i] /= x
	}
}
