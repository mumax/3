package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"reflect"
)

// Special buffered quantity to store magnetization
// makes sure it's normalized etc.
type magnetization struct {
	buffered
}

func (b *magnetization) Set(src *data.Slice) {
	if src.Mesh().Size() != b.buffer.Mesh().Size() {
		src = data.Resample(src, b.buffer.Mesh().Size())
	}
	data.Copy(b.buffer, src)
	cuda.Normalize(b.buffer, vol())
}

func (b *magnetization) LoadFile(fname string) {
	b.Set(LoadFile(fname))
}

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
