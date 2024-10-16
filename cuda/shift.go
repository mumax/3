package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// shift dst by shx cells (positive or negative) along X-axis.
// new edge value is clampL at left edge (-X) or clampR at right edge (+X).
func ShiftX(dst, src *data.Slice, shiftX int, clampL, clampR float32) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())
	N := dst.Size()
	cfg := make3DConf(N)
	k_shiftx_async(dst.DevPtr(0), src.DevPtr(0), N[X], N[Y], N[Z], shiftX, clampL, clampR, cfg)
}

// Shifts a component `src` of a vector field by `shiftX` cells along the X-axis.
// Unlike the normal `shift()`, the new edge value is the current edge value.
//
// To avoid the situation where the magnetization could be set to (0,0,0) within the geometry, it is
// also required to pass the two other vector components `othercomp` and `anothercomp` to this function.
// In cells where the vector (`src`, `othercomp`, `anothercomp`) is the zero-vector,
// `clampL` or `clampR` is used for the component `src` instead.
func ShiftEdgeCarryX(dst, src, othercomp, anothercomp *data.Slice, shiftX int, clampL, clampR float32) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1 && othercomp.NComp() == 1 && anothercomp.NComp() == 1)
	util.Assert(dst.Len() == src.Len())
	N := dst.Size()
	cfg := make3DConf(N)
	k_shiftedgecarryX_async(dst.DevPtr(0), src.DevPtr(0), othercomp.DevPtr(0), anothercomp.DevPtr(0), N[X], N[Y], N[Z], shiftX, clampL, clampR, cfg)
}

// shift dst by shy cells (positive or negative) along Y-axis.
// new edge value is clampD at bottom edge (-Y) or clampU at top edge (+Y)
func ShiftY(dst, src *data.Slice, shiftY int, clampD, clampU float32) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())
	N := dst.Size()
	cfg := make3DConf(N)
	k_shifty_async(dst.DevPtr(0), src.DevPtr(0), N[X], N[Y], N[Z], shiftY, clampD, clampU, cfg)
}

// Shifts a component `src` of a vector field by `shiftY` cells along the Y-axis.
// Unlike the normal `shift()`, the new edge value is the current edge value.
//
// To avoid the situation where the magnetization could be set to (0,0,0) within the geometry, it is
// also required to pass the two other vector components `othercomp` and `anothercomp` to this function.
// In cells where the vector (`src`, `othercomp`, `anothercomp`) is the zero-vector,
// `clampD` or `clampU` is used for the component `src` instead.
func ShiftEdgeCarryY(dst, src, othercomp, anothercomp *data.Slice, shiftY int, clampD, clampU float32) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1 && othercomp.NComp() == 1 && anothercomp.NComp() == 1)
	util.Assert(dst.Len() == src.Len())
	N := dst.Size()
	cfg := make3DConf(N)
	k_shiftedgecarryY_async(dst.DevPtr(0), src.DevPtr(0), othercomp.DevPtr(0), anothercomp.DevPtr(0), N[X], N[Y], N[Z], shiftY, clampD, clampU, cfg)
}

// shift dst by shz cells (positive or negative) along Z-axis.
// new edge value is clampB at back edge (-Z) or clampF at front edge (+Z).
func ShiftZ(dst, src *data.Slice, shiftZ int, clampB, clampF float32) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())
	N := dst.Size()
	cfg := make3DConf(N)
	k_shiftz_async(dst.DevPtr(0), src.DevPtr(0), N[X], N[Y], N[Z], shiftZ, clampB, clampF, cfg)
}

// Like Shift, but for bytes
func ShiftBytes(dst, src *Bytes, m *data.Mesh, shiftX int, clamp byte) {
	N := m.Size()
	cfg := make3DConf(N)
	k_shiftbytes_async(dst.Ptr, src.Ptr, N[X], N[Y], N[Z], shiftX, clamp, cfg)
}

func ShiftBytesY(dst, src *Bytes, m *data.Mesh, shiftY int, clamp byte) {
	N := m.Size()
	cfg := make3DConf(N)
	k_shiftbytesy_async(dst.Ptr, src.Ptr, N[X], N[Y], N[Z], shiftY, clamp, cfg)
}
