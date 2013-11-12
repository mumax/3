package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"reflect"
)

// Special buffered quantity to store magnetization
// makes sure it's normalized etc.
type magnetization struct {
	buffer *data.Slice
}

func (m *magnetization) Mesh() *data.Mesh { return Mesh() }
func (m *magnetization) NComp() int       { return 3 }
func (m *magnetization) Name() string     { return "m" }
func (m *magnetization) Unit() string     { return "" }

// allocate storage (not done by init, as mesh size may not yet be known then)
func (m *magnetization) alloc() {
	m.buffer = cuda.NewSlice(3, m.Mesh())
}

func (b *magnetization) Set(src *data.Slice) {
	if src.Mesh().Size() != b.buffer.Mesh().Size() {
		src = data.Resample(src, b.buffer.Mesh().Size())
	}
	data.Copy(b.buffer, src)
	cuda.Normalize(b.buffer, vol())
}

func (m *magnetization) LoadFile(fname string) {
	m.Set(LoadFile(fname))
}

func (m *magnetization) Slice() (s *data.Slice, recycle bool) {
	return m.buffer, false
}

func (m *magnetization) Region(r int) *sliceInRegion { return &sliceInRegion{m, r} }

func (m *magnetization) String() string { return util.Sprint(m.buffer.HostCopy()) }

// Set the value of one cell.
func (m *magnetization) SetCell(ix, iy, iz int, v ...float64) {
	nComp := m.NComp()
	util.Argument(len(v) == nComp)
	for c := 0; c < nComp; c++ {
		cuda.SetCell(m.buffer, c, ix, iy, iz, float32(v[c]))
	}
}

// Get the value of one cell.
func (m *magnetization) GetCell(comp, ix, iy, iz int) float64 {
	return float64(cuda.GetCell(m.buffer, comp, ix, iy, iz))
}

func (m *magnetization) TableData() []float64 { return Average(m) }

// Sets the magnetization inside the shape
func (m *magnetization) SetInShape(region Shape, conf Config) {
	if region == nil {
		region = universe
	}
	host := m.buffer.HostCopy()
	h := host.Vectors()
	n := m.Mesh().Size()

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				r := Index2Coord(ix, iy, iz)
				x, y, z := r[X], r[Y], r[Z]
				if region(x, y, z) { // inside
					m := conf(x, y, z)
					h[X][iz][iy][ix] = float32(m[X])
					h[Y][iz][iy][ix] = float32(m[Y])
					h[Z][iz][iy][ix] = float32(m[Z])
				}
			}
		}
	}
	m.Set(host)
}

// set m to config in region
func (m *magnetization) SetRegion(region int, conf Config) {
	host := m.buffer.HostCopy()
	h := host.Vectors()
	n := m.Mesh().Size()
	r := byte(region)

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				pos := Index2Coord(ix, iy, iz)
				x, y, z := pos[X], pos[Y], pos[Z]
				if regions.arr[iz][iy][ix] == r {
					m := conf(x, y, z)
					h[X][iz][iy][ix] = float32(m[X])
					h[Y][iz][iy][ix] = float32(m[Y])
					h[Z][iz][iy][ix] = float32(m[Z])
				}
			}
		}
	}
	m.Set(host)
}

func (m *magnetization) SetValue(v interface{})  { m.SetInShape(nil, v.(Config)) }
func (m *magnetization) InputType() reflect.Type { return reflect.TypeOf(Config(nil)) }
func (m *magnetization) Type() reflect.Type      { return reflect.TypeOf(new(magnetization)) }
func (m *magnetization) Eval() interface{}       { return m }
