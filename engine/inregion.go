package engine

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// represents a new quantity equal to q in the given region, 0 outside.
type sliceInRegion struct {
	slicer Slicer
	region int
}

func (q *sliceInRegion) NComp() int       { return q.slicer.NComp() }
func (q *sliceInRegion) Name() string     { return fmt.Sprint(q.slicer.Name(), ".region", q.region) }
func (q *sliceInRegion) Unit() string     { return q.slicer.Unit() }
func (q *sliceInRegion) Mesh() *data.Mesh { return q.slicer.Mesh() }
func (q *sliceInRegion) volume() float64  { return regions.volume(q.region) }

// returns a new slice equal to q in the given region, 0 outside.
func (q *sliceInRegion) Slice() (*data.Slice, bool) {
	src, r := q.slicer.Slice()
	if r {
		defer cuda.Recycle(src)
	}
	out := cuda.Buffer(q.NComp(), q.Mesh().Size())
	cuda.RegionSelect(out, src, regions.Gpu(), byte(q.region))
	return out, true
}

func (q *sliceInRegion) TableData() []float64 {
	slice, r := q.slicer.Slice()
	if r {
		defer cuda.Recycle(slice)
	}
	return averageRegion(slice, regions.volume(q.region))
}
