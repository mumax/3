package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Normalize vec to unit length, unless length or vol are zero.
func Normalize(vec, vol *data.Slice) {
	util.Argument(vol == nil || vol.NComp() == 1)
	N := vec.Len()
	cfg := make1DConf(N)
	k_normalize(vec.DevPtr(0), vec.DevPtr(1), vec.DevPtr(2), vol.DevPtr(0), N, cfg)
}
