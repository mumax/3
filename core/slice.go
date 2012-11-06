package core

import (
	"github.com/barnex/cuda5/safe"
)

type Slice struct {
	// ptr unsafe.Pointer
	// size [3]uint16
	// flag byte
	list []float32
	gpu  safe.Float32s
}

const(
	CPUACCESS = 1 << iota
	GPUACCESS
)

//func MakeSlice(ptr, size, flag)

func (s *Slice) Slice(a, b int) Slice {
	return Slice{s.list[a:b], s.gpu.Slice(a, b)}
}

func (s Slice) Host() []float32 {
	return s.list
}

func (s Slice) Gpu() safe.Float32s {
	return s.gpu
}
