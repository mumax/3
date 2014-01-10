package engine

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// represents a new quantity equal to q in the given region, 0 outside.
type oneRegion struct {
	parent Quantity
	region int
}

func (q *oneRegion) NComp() int       { return q.parent.NComp() }
func (q *oneRegion) Name() string     { return fmt.Sprint(q.parent.Name(), ".region", q.region) }
func (q *oneRegion) Unit() string     { return q.parent.Unit() }
func (q *oneRegion) Mesh() *data.Mesh { return q.parent.Mesh() }

// returns a new slice equal to q in the given region, 0 outside.
func (q *oneRegion) Slice() (*data.Slice, bool) {
	src, r := q.parent.Slice()
	if r {
		defer cuda.Recycle(src)
	}
	out := cuda.Buffer(q.NComp(), q.Mesh().Size())
	cuda.RegionSelect(out, src, regions.Gpu(), byte(q.region))
	return out, true
}

func (q *oneRegion) average() []float64 {
	slice, r := q.Slice()
	if r {
		defer cuda.Recycle(slice)
	}
	avg := sAverageUniverse(slice)
	sDiv(avg, regions.volume(q.region))
	return avg
}

func (q *oneRegion) Average() []float64 { return q.average() }

// slice division
func sDiv(v []float64, x float64) {
	for i := range v {
		v[i] /= x
	}
}
