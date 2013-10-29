package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"reflect"
)

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
// TODO: a bit slowish
func (m *magnetization) SetInShape(region Shape, conf Config) {
	if region == nil {
		region = universe
	}
	host := m.buffer.HostCopy()
	h := host.Vectors()
	n := m.Mesh().Size()

	for i := 0; i < n[0]; i++ {
		for j := 0; j < n[1]; j++ {
			for k := 0; k < n[2]; k++ {
				r := index2Coord(i, j, k)
				x, y, z := r[0], r[1], r[2]
				if region(x, y, z) { // inside
					m := conf(x, y, z)
					h[0][i][j][k] = float32(m[0])
					h[1][i][j][k] = float32(m[1])
					h[2][i][j][k] = float32(m[2])
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

	for i := 0; i < n[0]; i++ {
		for j := 0; j < n[1]; j++ {
			for k := 0; k < n[2]; k++ {
				pos := index2Coord(i, j, k)
				x, y, z := pos[0], pos[1], pos[2]
				if regions.arr[i][j][k] == r {
					m := conf(x, y, z)
					h[0][i][j][k] = float32(m[0])
					h[1][i][j][k] = float32(m[1])
					h[2][i][j][k] = float32(m[2])
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
