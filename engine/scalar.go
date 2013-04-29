package engine

import (
	"fmt"
)

type scalar struct {
	value     []float64
	timestamp int
	armed     bool
	autosave
	updateFn func() []float64
	// todo: deps: interface{arm}
}

func newScalar(name string, updateFn func() []float64) *scalar {
	s := new(scalar)
	s.name = name
	s.updateFn = updateFn
	return s
}

func (s *scalar) Get() []float64 {
	if s.timestamp != itime {
		fmt.Println("update", s.name) // debug. RM
		s.value = s.updateFn()
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
