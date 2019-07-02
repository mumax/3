package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func SetRxyPhiTheta(s *data.Slice, m *data.Slice) {
	N := s.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)
	k_setrxyphitheta_async(s.DevPtr(X), s.DevPtr(Y), s.DevPtr(Z), m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z), N[X], N[Y], N[Z], cfg)
	return
}
