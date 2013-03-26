package engine

import (
	"code.google.com/p/mx3/data"
)

type Handle interface {
}

type Buffered struct {
	s *data.Slice
}

type Adder struct {
	addFn func(dst *data.Slice) // calculates quantity and add result to dst
}
