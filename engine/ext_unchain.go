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
	DeclFunc("ext_Unchain", Unchain, "Switches a parameter from region-wise to per-cell values (currently only alpha). Every cell starts at its region's value.")
	DeclFunc("ext_Rechain", Rechain, "Switches an unchained parameter back to region-wise values, as they were before ext_Unchain.")
	DeclFunc("ext_SetParamInShape", SetParamInShape, "Sets an unchained parameter to a value inside a shape.")
	DeclFunc("ext_SetParamMask", SetParamMask, "Sets all per-cell values of an unchained parameter from a mask (NewScalarMask or LoadFile), resampled to the mesh if needed.")
}

func Unchain(p *RegionwiseScalar) {
	if p != Alpha {
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

func Rechain(p *RegionwiseScalar) {
	if p.cell == nil {
		return
	}
	cuda.Sync() // kernels may still read the buffer
	p.cell.Free()
	p.cell = nil
	p.invalidate()
}

func SetParamInShape(p *RegionwiseScalar, s Shape, v float64) {
	assertUnchained(p, "ext_SetParamInShape")
	if s == nil {
		s = universe
	}
	host := p.cellBuf().HostCopy()
	h := host.Scalars()
	n := Mesh().Size()
	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				r := Index2Coord(ix, iy, iz)
				if s(r[X], r[Y], r[Z]) {
					h[iz][iy][ix] = float32(v)
				}
			}
		}
	}
	data.Copy(p.cell, host)
}

func SetParamMask(p *RegionwiseScalar, mask *data.Slice) {
	assertUnchained(p, "ext_SetParamMask")
	if mask.NComp() != p.NComp() {
		util.Fatal("ext_SetParamMask: mask has ", mask.NComp(), " components, ", p.name, " needs ", p.NComp())
	}
	if mask.Size() != Mesh().Size() {
		mask = data.Resample(mask, Mesh().Size())
	}
	data.Copy(p.cellBuf(), mask)
}

func assertUnchained(p *RegionwiseScalar, fn string) {
	if p.cell == nil {
		util.Fatal(fn, ": ", p.name, " is region-wise, call ext_Unchain(", p.name, ") first")
	}
}

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
