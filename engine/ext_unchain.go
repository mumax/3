package engine

// Unchained parameters: per-cell values instead of the 256-entry region table.
// While unchained, SetRegion, GetRegion, plain assignment and time dependence
// are disabled; the region table is left untouched, so ext_Rechain restores it.

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func init() {
	DeclFunc("ext_Unchain", Unchain, "Switches a parameter from region-wise to per-cell values (currently alpha). Every cell starts at its region's value.")
	DeclFunc("ext_Rechain", Rechain, "Switches an unchained parameter back to region-wise values, as they were before ext_Unchain.")
	DeclFunc("ext_SetParamInShape", SetParamInShape, "Sets an unchained parameter to a value (scalar or vector) inside a shape.")
	DeclFunc("ext_SetParamMask", SetParamMask, "Sets all per-cell values of an unchained parameter from a mask (NewScalarMask, NewVectorMask, or LoadFile), resampled to the mesh if needed.")
}

// script-facing handle for RegionwiseScalar and RegionwiseVector
type unchainable interface{ rw() *regionwise }

func (p *regionwise) rw() *regionwise { return p }

// only params whose every use goes through MSlice/EvalTo/nonZero. Msat also reads the LUT directly (hasZero, lex2); needs more attention.
func canUnchain(p *regionwise) bool {
	return p == &Alpha.regionwise
}

func Unchain(param unchainable) {
	p := param.rw()
	if !canUnchain(p) {
		util.Fatal("ext_Unchain: only alpha can be unchained so far, not ", p.name)
	}
	if p.cell != nil {
		return
	}
	for _, f := range p.upd_reg {
		if f != nil {
			util.Fatal("ext_Unchain: ", p.name, " is time-dependent, set it to a constant first")
		}
	}
	checkMesh()
	p.cell = cuda.NewSlice(p.NComp(), Mesh().Size())
	p.lut.EvalTo(p.cell)
}

func Rechain(param unchainable) {
	p := param.rw()
	if p.cell == nil {
		return
	}
	cuda.Sync() // kernels may still read the buffer
	p.cell.Free()
	p.cell = nil
	p.invalidate()
}

// v: number for a scalar parameter, vector(...) for a vector parameter
func SetParamInShape(param unchainable, s Shape, v interface{}) {
	p := param.rw()
	assertUnchained(p, "ext_SetParamInShape")
	var val []float64
	switch v := v.(type) {
	case float64:
		val = []float64{v}
	case int:
		val = []float64{float64(v)}
	case data.Vector:
		val = v[:]
	}
	if len(val) != p.NComp() {
		util.Fatal("ext_SetParamInShape: ", p.name, " needs ", p.NComp(), " component(s), got ", v)
	}
	if s == nil {
		s = universe
	}
	host := p.cellBuf().HostCopy()
	n := Mesh().Size()
	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				r := Index2Coord(ix, iy, iz)
				if s(r[X], r[Y], r[Z]) {
					for c := range val {
						host.Set(c, ix, iy, iz, val[c])
					}
				}
			}
		}
	}
	data.Copy(p.cell, host)
}

func SetParamMask(param unchainable, mask *data.Slice) {
	p := param.rw()
	assertUnchained(p, "ext_SetParamMask")
	if mask.NComp() != p.NComp() {
		util.Fatal("ext_SetParamMask: mask has ", mask.NComp(), " components, ", p.name, " needs ", p.NComp())
	}
	if mask.Size() != Mesh().Size() {
		mask = data.Resample(mask, Mesh().Size())
	}
	data.Copy(p.cellBuf(), mask)
}

func assertUnchained(p *regionwise, fn string) {
	if p.cell == nil {
		util.Fatal(fn, ": ", p.name, " is region-wise, call ext_Unchain(", p.name, ") first")
	}
}

// unchained values are assumed non-zero: skips a GPU reduction per step, zero cells just add zero
func (p *regionwise) isZero() bool  { return p.cell == nil && p.lut.isZero() }
func (p *regionwise) nonZero() bool { return !p.isZero() }

func (p *regionwise) unchained() bool { return p.cell != nil }

// per-cell buffer, resampled if the mesh changed since ext_Unchain
func (p *regionwise) cellBuf() *data.Slice {
	if p.cell.Size() != Mesh().Size() {
		host := data.Resample(p.cell.HostCopy(), Mesh().Size())
		cuda.Sync()
		p.cell.Free()
		p.cell = cuda.NewSlice(p.NComp(), Mesh().Size())
		data.Copy(p.cell, host)
	}
	return p.cell
}
