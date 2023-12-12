package cuda

import (
	"github.com/mumax/3/v3/data"
	"github.com/mumax/3/v3/util"
)

// shift dst by shx cells (positive or negative) along X-axis.
// new edge value is clampL at left edge or clampR at right edge.
func ShiftX(dst, src *data.Slice, shiftX int, clampL, clampR float32) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())
	N := dst.Size()
	cfg := make3DConf(N)
	k_shiftx_async(dst.DevPtr(0), src.DevPtr(0), N[X], N[Y], N[Z], shiftX, clampL, clampR, cfg)
}

func ShiftY(dst, src *data.Slice, shiftY int, clampL, clampR float32) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())
	N := dst.Size()
	cfg := make3DConf(N)
	k_shifty_async(dst.DevPtr(0), src.DevPtr(0), N[X], N[Y], N[Z], shiftY, clampL, clampR, cfg)
}

func ShiftZ(dst, src *data.Slice, shiftZ int, clampL, clampR float32) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())
	N := dst.Size()
	cfg := make3DConf(N)
	k_shiftz_async(dst.DevPtr(0), src.DevPtr(0), N[X], N[Y], N[Z], shiftZ, clampL, clampR, cfg)
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
