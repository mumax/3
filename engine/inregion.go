package engine

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// constrains a Getter to a region
type inRegion struct {
	q      Getter
	region int
}

func (q *inRegion) NComp() int       { return q.q.NComp() }
func (q *inRegion) Name() string     { return fmt.Sprint(q.q.Name(), ".region", q.region) }
func (q *inRegion) Unit() string     { return q.q.Unit() }
func (q *inRegion) Mesh() *data.Mesh { return q.q.Mesh() }

func (q *inRegion) Get() (*data.Slice, bool) {
	src, r := q.q.Get()
	if r {
		defer cuda.Recycle(src)
	}
	out := cuda.Buffer(q.NComp(), q.Mesh())
	cuda.RegionSelect(out, src, regions.Gpu(), byte(q.region))
	return out, true
}

func (q *inRegion) TableData() []float64 {
	buf, r := q.Get()
	if r {
		defer cuda.Recycle(buf)
	}
	return avg(buf)
}

// constrains inputParam to a region
type selectRegion struct {
	*inputParam
	region int
}

func (p *selectRegion) TableData() []float64 {
	return p.getRegion(p.region)
}
