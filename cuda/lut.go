package cuda

import (
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

const LUTSIZE = 256

// Look-up table maps region numbers to values
type LUT struct {
	gpu unsafe.Pointer   // gpu copy of cpu
	cpu [LUTSIZE]float32 //
}

func (l *LUT) Init() {
	l.gpu = MemAlloc(LUTSIZE * cu.SIZEOF_FLOAT32)
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

type LUTs []LUT

func NewLUTs(nComp int) LUTs {
	l := make(LUTs, nComp)
	for c := range l {
		l[c].Init()
	}
	return l
}

func (l *LUTs) Ptr(comp int) unsafe.Pointer { return (*l)[comp].gpu }
