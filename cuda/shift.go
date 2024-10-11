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

// shift dst by shx cells (positive or negative) along X-axis.
// new edge value is the current value at the border.
func ShiftEdgeCarryX(dst, src *data.Slice, shiftX int) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())
	N := dst.Size()
	cfg := make3DConf(N)
	k_shiftedgecarryX_async(dst.DevPtr(0), src.DevPtr(0), N[X], N[Y], N[Z], shiftX, cfg)
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

// shift dst by shy cells (positive or negative) along Y-axis.
// new edge value is the current value at the border.
func ShiftEdgeCarryY(dst, src *data.Slice, shiftY int) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())
	N := dst.Size()
	cfg := make3DConf(N)
	k_shiftedgecarryY_async(dst.DevPtr(0), src.DevPtr(0), N[X], N[Y], N[Z], shiftY, cfg)
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
