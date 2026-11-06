package engine

// Calculation of stray field

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/util"
)

var (
	strayconv_      *cuda.StrayFieldConvolution
	strayconvoffset [3]float64
)

func init() {
	DeclFunc("GetStrayField", GetStrayField, "Get the stray field at an offset w.r.t. each cell of the grid.")
}

type strayField struct {
	slice  *data.Slice
	mesh   *data.Mesh
	offset [3]float64
}

func (s *strayField) NComp() int {
	return s.slice.NComp()
}

func (s *strayField) EvalTo(dst *data.Slice) {
	util.Assert(dst.NComp() == s.NComp())
	data.Copy(dst, s.slice)
}

func (s *strayField) Mesh() *data.Mesh {
	return s.mesh
}

func (s *strayField) Comp(i int) *strayField {
	return NewStrayField(s.slice.Comp(i), s.Mesh().Size(), s.offset)
}

func NewStrayField(slice *data.Slice, size [3]int, offset [3]float64) *strayField {
	cs := Mesh().CellSize() // Stray field requires same cell size as the main mesh
	pbc := Mesh().PBC()     // Stray field follows same PBC as main mesh
	mesh := data.NewMesh(size[X], size[Y], size[Z], cs[X], cs[Y], cs[Z], pbc[X], pbc[Y], pbc[Z])
	return &strayField{slice, mesh, offset}
}

func GetStrayField(dx, dy, dz float64) *strayField {
	size := Mesh().Size() // TODO: allow arbitrary sizes defined by e.g. B_stray_corner1 and B_stray_corner2
	if dx != strayconvoffset[X] || dy != strayconvoffset[Y] || dz != strayconvoffset[Z] {
		strayconvoffset = [3]float64{dx, dy, dz}
		reinitstrayconv()
	}

	gpu := cuda.Buffer(VECTOR, size)
	defer cuda.Recycle(gpu)

	SetStrayField(gpu)

	host := data.NewSlice(VECTOR, size)
	data.Copy(host, gpu)

	return NewStrayField(host, size, strayconvoffset)
}

// Sets dst to the current demag field at an offset from each cell.
// Does not care about EnableDemag. dst requires GPUAccess.
func SetStrayField(dst *data.Slice) {
	msat := Msat.MSlice()
	defer msat.Recycle()

	geom, r := geometry.Slice()
	if r {
		defer cuda.Recycle(geom)
	}
	var maskedgeom *data.Slice
	if NoDemagSpins.isZero() {
		// Normal demag, everywhere
		maskedgeom = geom
	} else {
		// No-demag spins: mask-out geometry with zeros where NoDemagSpins is set,
		// so these spins do not generate a field
		maskedgeom = cuda.Buffer(SCALAR, geom.Size()) // masked-out geometry
		defer func() { cuda.Recycle(maskedgeom) }()

		// obtain a copy of the geometry mask, which we can overwrite
		data.Copy(maskedgeom, geom)

		// mask-out
		cuda.ZeroMask(maskedgeom, NoDemagSpins.gpuLUT1(), regions.Gpu())
	}

	// Set dst to the demag field, but cells where NoDemagSpins != 0 do not generate field.
	strayConv().Exec(dst, M.Buffer(), maskedgeom, msat)
}

// returns stray field convolution, making sure it's initialized
func strayConv() *cuda.StrayFieldConvolution {
	if strayconv_ == nil {
		reinitstrayconv()
	}
	return strayconv_
}

func reinitstrayconv() {
	SetBusy(true)
	defer SetBusy(false)
	if strayconv_ != nil {
		strayconv_.Free()
		strayconv_ = nil
	}
	kernel := mag.StrayKernel(Mesh().Size(), Mesh().PBC(), Mesh().CellSize(), DemagAccuracy, strayconvoffset, *Flag_cachedir)
	strayconv_ = cuda.NewStray(Mesh().Size(), Mesh().PBC(), kernel)
}
