package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"math"
)

// special bufferedQuant to store magnetization.
// Set is overridden to stencil m with geometry.
type magnetization struct {
	bufferedQuant
}

func (q *magnetization) init() {
	q.bufferedQuant = buffered(cuda.NewSlice(3, Mesh()), "m", "")
}

// overrides normal set to allow stencil ops
func (b *magnetization) Set(src *data.Slice) {
	if src.Mesh().Size() != b.buffer.Mesh().Size() {
		src = data.Resample(src, b.buffer.Mesh().Size())
	}
	stencil(src, vol.host)
	data.Copy(b.buffer, src)
}

func (m *magnetization) setRegion(conf Config, region Shape) {
	if region == nil {
		region = universe
	}

	host := hostBuf(3, m.Mesh())
	data.Copy(host, m.buffer)

	h := host.Vectors()
	n := m.Mesh().Size()
	c := m.Mesh().CellSize()
	dx := (float64(n[2]/2) - 0.5) * c[2]
	dy := (float64(n[1]/2) - 0.5) * c[1]
	dz := (float64(n[0]/2) - 0.5) * c[0]
	stencil := vol.host.Scalars()

	for i := 0; i < n[0]; i++ {
		z := float64(i)*c[0] - dz
		for j := 0; j < n[1]; j++ {
			y := float64(j)*c[1] - dy
			for k := 0; k < n[2]; k++ {
				x := float64(k)*c[2] - dx

				sten := stencil[i][j][k]
				if sten == 0 {
					h[0][i][j][k] = 0
					h[1][i][j][k] = 0
					h[2][i][j][k] = 0
				} else {
					inside := region(x, y, z)
					if inside {
						m := normalize(conf(x, y, z))
						h[0][i][j][k] = float32(m[2]) * sten
						h[1][i][j][k] = float32(m[1]) * sten
						h[2][i][j][k] = float32(m[0]) * sten
					}
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
func (m *magnetization) stencil(s *data.Slice) {
	if s.IsNil() {
		return
	}
	h := hostBuf(m.NComp(), m.Mesh())
	data.Copy(h, m.buffer)
	stencil(h, s)
	data.Copy(m.buffer, h)
}

// remove dst where stencil is zero (host).
func stencil(dst, stencil *data.Slice) {
	if stencil.IsNil() {
		return
	}
	util.Argument(stencil.NComp() == 1)
	s := stencil.Host()[0]
	d := dst.Host()
	for c := range d {
		for i := range d[c] {
			d[c][i] *= s[i]
		}
	}
}

// TODO: normalize M after set
