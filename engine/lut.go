package engine

import (
	"code.google.com/p/mx3/cuda"
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

const LUTSIZE = 256

type LUT struct {
	cpu [LUTSIZE]float32
	gpu unsafe.Pointer
}

func (l *LUT) init() {
	l.gpu = cuda.MemAlloc(LUTSIZE * cu.SIZEOF_FLOAT32)
	cu.MemsetD32(cu.DevicePtr(l.gpu), 0, LUTSIZE)
}

func (l *LUT) Set(region int, value float32) {
	l.cpu[region] = value
	l.upload()
}

func (l *LUT) SetAllRegions(value float32) {
	for i := range l.cpu {
		l.cpu[i] = value
	}
	l.upload()
}

func (l *LUT) upload() {
	cu.MemcpyHtoD(cu.DevicePtr(l.gpu), unsafe.Pointer(&l.cpu[0]), cu.SIZEOF_FLOAT32*LUTSIZE)
}

func (l *LUT) Cell(i, j, k int) float32 {
	return l.cpu[regions.arr[i][j][k]]
}
