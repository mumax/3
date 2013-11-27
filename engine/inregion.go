package engine

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// represents a new quantity equal to q in the given region, 0 outside.
type sliceInRegion struct {
	q      Slicer
	region int
}

func (q *sliceInRegion) NComp() int                 { return q.q.NComp() }
func (q *sliceInRegion) Name() string               { return fmt.Sprint(q.q.Name(), ".region", q.region) }
func (q *sliceInRegion) Unit() string               { return q.q.Unit() }
func (q *sliceInRegion) Mesh() *data.Mesh           { return q.q.Mesh() }
func (q *sliceInRegion) Slice() (*data.Slice, bool) { return getRegion(q.q, q.region) }
func (q *sliceInRegion) TableData() []float64       { return Average(q) }
func (q *sliceInRegion) volume() float64            { return regions.volume(q.region) }

//// represents a new quantity equal to q in the given region, 0 outside.
//type paramInRegion struct {
//	q      *inputParam
//	region int
//}
//
//func (p *paramInRegion) NComp() int                 { return p.q.NComp() }
//func (p *paramInRegion) Name() string               { return fmt.Sprint(p.q.Name(), ".region", p.region) }
//func (p *paramInRegion) Unit() string               { return p.q.Unit() }
//func (p *paramInRegion) Mesh() *data.Mesh           { return p.q.Mesh() }
//func (p *paramInRegion) Slice() (*data.Slice, bool) { return getRegion(p.q, p.region) }
//func (p *paramInRegion) TableData() []float64       { return Average(q) }
//func (p *paramInRegion) volume() float64            { return regions.volume(p.region) }

// returns a new slice equal to q in the given region, 0 outside.
func getRegion(q Slicer, region int) (*data.Slice, bool) {
	src, r := q.Slice()
	if r {
		defer cuda.Recycle(src)
	}
	out := cuda.Buffer(q.NComp(), q.Mesh().Size())
	cuda.RegionSelect(out, src, regions.Gpu(), byte(region))
	return out, true
}
