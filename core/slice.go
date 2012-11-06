package core

import (
	"github.com/barnex/cuda5/safe"
)

type Slice struct {
	list []float32
	gpu safe.Float32s
}

func (s *Slice) Slice(a, b int) Slice {
	return Slice{s.list[a:b], s.gpu.Slice(a, b)}
}

func (s Slice) Host() []float32 {
	return s.list
}

func (s Slice) Gpu() safe.Float32s {
	return s.gpu
}
