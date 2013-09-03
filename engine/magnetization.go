package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"math"
	"reflect"
)

// special bufferedQuant to store magnetization.
// Set is overridden to stencil m with geometry.
type magnetization struct {
	bufferedQuant
}

func (m *magnetization) SetValue(v interface{})  { m.SetInShape(nil, v.(Config)) }
func (m *magnetization) InputType() reflect.Type { return reflect.TypeOf(Config(nil)) }
func (m *magnetization) Type() reflect.Type      { return reflect.TypeOf(new(magnetization)) }
func (m *magnetization) Eval() interface{}       { return m }

func (q *magnetization) init() {
	q.bufferedQuant = buffered(cuda.NewSlice(3, Mesh()), "m", "")
}

// overrides normal set to allow stencil ops
func (b *magnetization) Set(src *data.Slice) {
	if src.Mesh().Size() != b.buffer.Mesh().Size() {
		src = data.Resample(src, b.buffer.Mesh().Size())
	}
	//stencil(src, vol.host) // TODO: stencil !!
	data.Copy(b.buffer, src)
}

// Read a magnetization state from .dump file.
func (b *magnetization) LoadFile(fname string) {
	s, _ := data.MustReadFile(fname)
	b.Set(s)
}

// Sets the magnetization inside the shape
func (m *magnetization) SetInShape(region Shape, conf Config) {
	if region == nil {
		region = universe
	}

	host := m.buffer.HostCopy()
	h := host.Vectors()
	n := m.Mesh().Size()
	c := m.Mesh().CellSize()
	dx := (float64(n[2]/2) - 0.5) * c[2]
	dy := (float64(n[1]/2) - 0.5) * c[1]
	dz := (float64(n[0]/2) - 0.5) * c[0]

	for i := 0; i < n[0]; i++ {
		z := float64(i)*c[0] - dz
		for j := 0; j < n[1]; j++ {
			y := float64(j)*c[1] - dy
			for k := 0; k < n[2]; k++ {
				x := float64(k)*c[2] - dx

				sten := regions.arr[i][j][k]
				if sten == 0 {
					h[0][i][j][k] = 0
					h[1][i][j][k] = 0
					h[2][i][j][k] = 0
				} else if region(x, y, z) { // inside
					m := normalize(conf(x, y, z))
					h[0][i][j][k] = float32(m[2])
					h[1][i][j][k] = float32(m[1])
					h[2][i][j][k] = float32(m[0])
				}
			}
		}
	}
	data.Copy(m.buffer, host)
}

func normalize(v [3]float64) [3]float64 {
	s := 1 / math.Sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
	return [3]float64{s * v[0], s * v[1], s * v[2]}
}

// remove magnetization where stencil is zero.
func (m *magnetization) stencilGeom() {
	if geom == nil {
		return
	}
	h := m.buffer.HostCopy()
	stencil(h, regions.cpu)
	data.Copy(m.buffer, h)
}

// remove dst where stencil is zero (host).
func stencil(dst *data.Slice, stencil []byte) {
	d := dst.Host()
	for i, s := range stencil {
		if s == 0 {
			d[0][i] = 0
			d[1][i] = 0
			d[2][i] = 0
		}
	}
}

// TODO: normalize M after set
