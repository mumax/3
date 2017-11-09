package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math/rand"
)

func init() {
	DeclFunc("SetGeom", SetGeom, "Sets the geometry to a given shape")
	DeclVar("EdgeSmooth", &edgeSmooth, "Geometry edge smoothing with edgeSmooth^3 samples per cell, 0=staircase, ~8=very smooth")
	geometry.init()
}

var (
	geometry   geom
	edgeSmooth int = 0 // disabled by default
)

type geom struct {
	info
	buffer *data.Slice
	shape  Shape
}

func (g *geom) init() {
	g.buffer = nil
	g.info = info{1, "geom", ""}
	DeclROnly("geom", &geometry, "Cell fill fraction (0..1)")
}

func spaceFill() float64 {
	if geometry.Gpu().IsNil() {
		return 1
	} else {
		return float64(cuda.Sum(geometry.buffer)) / float64(geometry.Mesh().NCell())
	}
}

func (g *geom) Gpu() *data.Slice {
	if g.buffer == nil {
		g.buffer = data.NilSlice(1, g.Mesh().Size())
	}
	return g.buffer
}

func (g *geom) Slice() (*data.Slice, bool) {
	s := g.Gpu()
	if s.IsNil() {
		s := cuda.Buffer(g.NComp(), g.Mesh().Size())
		cuda.Memset(s, 1)
		return s, true
	} else {
		return s, false
	}
}

func (q *geom) EvalTo(dst *data.Slice) { EvalTo(q, dst) }

var _ Quantity = &geometry

func (g *geom) average() []float64 {
	s, r := g.Slice()
	if r {
		defer cuda.Recycle(s)
	}
	return sAverageUniverse(s)
}

func (g *geom) Average() float64 { return g.average()[0] }

func SetGeom(s Shape) {
	geometry.setGeom(s)
}

func (geometry *geom) setGeom(s Shape) {
	SetBusy(true)
	defer SetBusy(false)

	if s == nil {
		// TODO: would be nice not to save volume if entirely filled
		s = universe
	}

	geometry.shape = s
	if geometry.Gpu().IsNil() {
		geometry.buffer = cuda.NewSlice(1, geometry.Mesh().Size())
	}

	host := data.NewSlice(1, geometry.Gpu().Size())
	array := host.Scalars()
	V := host
	v := array
	n := geometry.Mesh().Size()
	c := geometry.Mesh().CellSize()
	cx, cy, cz := c[X], c[Y], c[Z]

	progress, progmax := 0, n[Y]*n[Z]

	var ok bool
	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {

			progress++
			util.Progress(progress, progmax, "Initializing geometry")

			for ix := 0; ix < n[X]; ix++ {

				r := Index2Coord(ix, iy, iz)
				x0, y0, z0 := r[X], r[Y], r[Z]

				// check if center and all vertices lie inside or all outside
				allIn, allOut := true, true
				if s(x0, y0, z0) {
					allOut = false
				} else {
					allIn = false
				}

				if edgeSmooth != 0 { // center is sufficient if we're not really smoothing
					for _, Δx := range []float64{-cx / 2, cx / 2} {
						for _, Δy := range []float64{-cy / 2, cy / 2} {
							for _, Δz := range []float64{-cz / 2, cz / 2} {
								if s(x0+Δx, y0+Δy, z0+Δz) { // inside
									allOut = false
								} else {
									allIn = false
								}
							}
						}
					}
				}

				switch {
				case allIn:
					v[iz][iy][ix] = 1
					ok = true
				case allOut:
					v[iz][iy][ix] = 0
				default:
					v[iz][iy][ix] = geometry.cellVolume(ix, iy, iz)
					ok = ok || (v[iz][iy][ix] != 0)
				}
			}
		}
	}

	if !ok {
		util.Fatal("SetGeom: geometry completely empty")
	}

	data.Copy(geometry.buffer, V)

	// M inside geom but previously outside needs to be re-inited
	needupload := false
	geomlist := host.Host()[0]
	mhost := M.Buffer().HostCopy()
	m := mhost.Host()
	rng := rand.New(rand.NewSource(0))
	for i := range m[0] {
		if geomlist[i] != 0 {
			mx, my, mz := m[X][i], m[Y][i], m[Z][i]
			if mx == 0 && my == 0 && mz == 0 {
				needupload = true
				rnd := randomDir(rng)
				m[X][i], m[Y][i], m[Z][i] = float32(rnd[X]), float32(rnd[Y]), float32(rnd[Z])
			}
		}
	}
	if needupload {
		data.Copy(M.Buffer(), mhost)
	}

	M.normalize() // removes m outside vol
}

// Sample edgeSmooth^3 points inside the cell to estimate its volume.
func (g *geom) cellVolume(ix, iy, iz int) float32 {
	r := Index2Coord(ix, iy, iz)
	x0, y0, z0 := r[X], r[Y], r[Z]

	c := geometry.Mesh().CellSize()
	cx, cy, cz := c[X], c[Y], c[Z]
	s := geometry.shape
	var vol float32

	N := edgeSmooth
	S := float64(edgeSmooth)

	for dx := 0; dx < N; dx++ {
		Δx := -cx/2 + (cx / (2 * S)) + (cx/S)*float64(dx)
		for dy := 0; dy < N; dy++ {
			Δy := -cy/2 + (cy / (2 * S)) + (cy/S)*float64(dy)
			for dz := 0; dz < N; dz++ {
				Δz := -cz/2 + (cz / (2 * S)) + (cz/S)*float64(dz)

				if s(x0+Δx, y0+Δy, z0+Δz) { // inside
					vol++
				}
			}
		}
	}
	return vol / float32(N*N*N)
}

func (g *geom) shift(dx int) {
	// empty mask, nothing to do
	if g == nil || g.buffer.IsNil() {
		return
	}

	// allocated mask: shift
	s := g.buffer
	s2 := cuda.Buffer(1, g.Mesh().Size())
	defer cuda.Recycle(s2)
	newv := float32(1) // initially fill edges with 1's
	cuda.ShiftX(s2, s, dx, newv, newv)
	data.Copy(s, s2)

	n := Mesh().Size()
	x1, x2 := shiftDirtyRange(dx)

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := x1; ix < x2; ix++ {
				r := Index2Coord(ix, iy, iz) // includes shift
				if !g.shape(r[X], r[Y], r[Z]) {
					cuda.SetCell(g.buffer, 0, ix, iy, iz, 0) // a bit slowish, but hardly reached
				}
			}
		}
	}

}

// x range that needs to be refreshed after shift over dx
func shiftDirtyRange(dx int) (x1, x2 int) {
	nx := Mesh().Size()[X]
	util.Argument(dx != 0)
	if dx < 0 {
		x1 = nx + dx
		x2 = nx
	} else {
		x1 = 0
		x2 = dx
	}
	return
}


/// Y geom shift Victor 07/10/2016

func (g *geom) shiftY(dy int) {
	// empty mask, nothing to do
	if g == nil || g.buffer.IsNil() {
		return
	}

	// allocated mask: shift
	s := g.buffer
	s2 := cuda.Buffer(1, g.Mesh().Size())
	defer cuda.Recycle(s2)
	newv := float32(1) // initially fill edges with 1's
	cuda.ShiftY(s2, s, dy, newv, newv)
	data.Copy(s, s2)

	n := Mesh().Size()
	y1, y2 := shiftDirtyRangeY(dy)

	for iz := 0; iz < n[Z]; iz++ {
		for ix := 0; ix < n[X]; ix++ {
			for iy := y1; iy < y2; iy++ {
				r := Index2Coord(ix, iy, iz) // includes shift
				if !g.shape(r[X], r[Y], r[Z]) {
					cuda.SetCell(g.buffer, 0, ix, iy, iz, 0) // a bit slowish, but hardly reached
				}
			}
		}
	}

}

// x range that needs to be refreshed after shift over dx
func shiftDirtyRangeY(dy int) (y1, y2 int) {
	ny := Mesh().Size()[Y]
	util.Argument(dy != 0)
	if dy < 0 {
		y1 = ny + dy
		y2 = ny
	} else {
		y1 = 0
		y2 = dy
	}
	return
}




func (g *geom) Mesh() *data.Mesh { return Mesh() }
