package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"reflect"
)

// A buffered quantity is stored in GPU memory at all times.
type buffered struct {
	info
	buffer *data.Slice
}

// init metadata but does not allocate yet
func (b *buffered) init(nComp int, name, unit, doc_ string, mesh *data.Mesh) {
	b.info = Info(nComp, name, unit, mesh)
	DeclLValue(name, b, doc_)
}

// allocate storage (not done by init)
func (q *buffered) alloc() {
	q.buffer = cuda.NewSlice(3, q.Mesh())
}

// get buffer (on GPU, no need to recycle)
func (b *buffered) Get() (q *data.Slice, recycle bool) {
	return b.buffer, false
}

// Set the value of one cell.
func (b *buffered) SetCell(ix, iy, iz int, v ...float64) {
	nComp := b.NComp()
	util.Argument(len(v) == nComp)
	for c := 0; c < nComp; c++ {
		cuda.SetCell(b.buffer, util.SwapIndex(c, nComp), iz, iy, ix, float32(v[c]))
	}
}

// Get the value of one cell.
func (b *buffered) GetCell(comp, ix, iy, iz int) float64 {
	return float64(cuda.GetCell(b.buffer, util.SwapIndex(comp, b.NComp()), iz, iy, ix))
}

// overrides normal set to allow stencil ops
func (b *buffered) Set(src *data.Slice) {
	if src.Mesh().Size() != b.buffer.Mesh().Size() {
		src = data.Resample(src, b.buffer.Mesh().Size())
	}
	//stencil(src, vol.host) // TODO: stencil !!
	data.Copy(b.buffer, src)
}

func (b *buffered) LoadFile(fname string) {
	b.Set(LoadFile(fname))
}

// Shift the data over (shx, shy, shz cells), clamping boundary values.
// Typically used in a PostStep function to center the magnetization on
// the simulation window.
func (b *buffered) Shift(shx, shy, shz int) {
	m2 := cuda.Buffer(1, b.buffer.Mesh())
	defer cuda.Recycle(m2)
	for c := 0; c < b.NComp(); c++ {
		comp := b.buffer.Comp(c)
		cuda.Shift(m2, comp, [3]int{shz, shy, shx}) // ZYX !
		data.Copy(comp, m2)
	}
}

// Sets the magnetization inside the shape
// TODO: a bit slowish
func (m *buffered) SetInShape(region Shape, conf Config) {
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
				if region(x, y, z) { // inside
					m := conf(x, y, z)
					h[0][i][j][k] = float32(m[2])
					h[1][i][j][k] = float32(m[1])
					h[2][i][j][k] = float32(m[0])
				}
			}
		}
	}
	data.Copy(m.buffer, host)
	cuda.Normalize(M.buffer, vol) // removes m outside vol
}

// set m to config in region
func (m *buffered) SetRegion(region int, conf Config) {
	host := m.buffer.HostCopy()
	h := host.Vectors()
	n := m.Mesh().Size()
	c := m.Mesh().CellSize()
	dx := (float64(n[2]/2) - 0.5) * c[2]
	dy := (float64(n[1]/2) - 0.5) * c[1]
	dz := (float64(n[0]/2) - 0.5) * c[0]
	r := byte(region)

	for i := 0; i < n[0]; i++ {
		z := float64(i)*c[0] - dz
		for j := 0; j < n[1]; j++ {
			y := float64(j)*c[1] - dy
			for k := 0; k < n[2]; k++ {
				x := float64(k)*c[2] - dx

				if regions.arr[i][j][k] == r {
					m := conf(x, y, z)
					h[0][i][j][k] = float32(m[2])
					h[1][i][j][k] = float32(m[1])
					h[2][i][j][k] = float32(m[0])
				}
			}
		}
	}
	data.Copy(m.buffer, host)
	cuda.Normalize(m.buffer, vol)
}

func (m *buffered) SetValue(v interface{})  { m.SetInShape(nil, v.(Config)) } // TODO: SetInShape a bit slowish
func (m *buffered) InputType() reflect.Type { return reflect.TypeOf(Config(nil)) }
func (m *buffered) Type() reflect.Type      { return reflect.TypeOf(new(buffered)) }
func (m *buffered) Eval() interface{}       { return m }
