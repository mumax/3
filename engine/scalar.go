package engine

type scalar struct {
	nComp      int
	name, unit string
	value      []float64
	timestamp  int
	armed      bool
	updateFn   func() []float64 // need on-the-fly and from-zero
	UpdCount   int
}

func newScalar(nComp int, name, unit string, updateFn func() []float64) *scalar {
	s := new(scalar)
	s.nComp = nComp
	s.name = name
	s.unit = unit
	s.updateFn = updateFn
	return s
}

func (s *scalar) Get() []float64 {
	if s.timestamp != itime {
		s.value = s.updateFn()
		s.UpdCount++
		s.timestamp = itime
	}
	return s.value
}

func (s *scalar) touch(good bool) {
	if s.armed {
		_ = s.Get() // update s.value
		s.armed = false
	}
}

// when armed, updateFn will fire upon next touch
func (s *scalar) arm() {
	s.armed = true
}
