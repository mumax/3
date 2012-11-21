package gpu

//func NewExchange6(tag string, m nimble.ChanN, aex_reduced float64) *Stencil{
//
//}

//import (
//	"github.com/barnex/cuda5/cu"
//	"github.com/barnex/cuda5/safe"
//	"nimble-cube/core"
//	"nimble-cube/gpu/ptx"
//	"unsafe"
//)
//
//type Exchange6 struct {
//	m   core.RChan3
//	hex core.Chan3
//	*core.Mesh
//	aex_reduced float64
//	factors     [3]float32
//	stream      cu.Stream
//}
//
//func RunExchange6(tag string, m_ core.Chan3, aex_reduced float64) *Exchange6 {
//	m := m_.NewReader()
//	mesh := m.Mesh()
//	hex := MakeChan3(tag, "T", mesh) //TODO: blocks
//	e := &Exchange6{m: m, hex: hex, Mesh: mesh, aex_reduced: aex_reduced, stream: cu.StreamCreate()}
//	cellsize := mesh.CellSize()
//	for i := range e.factors {
//		e.factors[i] = float32(aex_reduced / (cellsize[i] * cellsize[i]))
//	}
//	core.Stack(e)
//	return e
//}
//
//func (e *Exchange6) Output() core.Chan3 { return e.hex }
//
//func (e *Exchange6) Run() {
//	LockCudaThread()
//	N := e.NCell()
//	for {
//		m := e.m.ReadNext(N)
//		hex := e.hex.WriteNext(N)
//		for i := 0; i < 3; i++ {
//			exchange6(hex[i], m[i], e.Mesh, e.factors, e.stream)
//		}
//		e.hex.WriteDone()
//		e.m.ReadDone()
//	}
//}
//
//var exchange6Code cu.Function
//
////__global__ void exchange6(float* h, float* m, float fac0, float fac1, float fac2, int wrap0, int wrap1, int wrap2, int N0, int N1, int N2){
//func exchange6(h, m safe.Float32s, mesh *core.Mesh, factors [3]float32, stream cu.Stream) {
//	core.Assert(h.Len() == m.Len() && h.Len() == mesh.NCell())
//
//	if exchange6Code == 0 {
//		mod := cu.ModuleLoadData(ptx.EXCHANGE6)
//		exchange6Code = mod.GetFunction("exchange6")
//	}
//
//	size := mesh.Size()
//	N0, N1, N2 := size[0], size[1], size[2]
//	wrap := mesh.PBC()
//
//	gridDim, blockDim := Make2DConf(N2, N1) // why?
//
//	hptr := h.Pointer()
//	mptr := m.Pointer()
//
//	args := []unsafe.Pointer{
//		unsafe.Pointer(&hptr),
//		unsafe.Pointer(&mptr),
//		unsafe.Pointer(&factors[0]),
//		unsafe.Pointer(&factors[1]),
//		unsafe.Pointer(&factors[2]),
//		unsafe.Pointer(&wrap[0]),
//		unsafe.Pointer(&wrap[1]),
//		unsafe.Pointer(&wrap[2]),
//		unsafe.Pointer(&N0),
//		unsafe.Pointer(&N1),
//		unsafe.Pointer(&N2)}
//
//	shmem := 0
//	cu.LaunchKernel(exchange6Code, gridDim.X, gridDim.Y, gridDim.Z, blockDim.X, blockDim.Y, blockDim.Z, shmem, stream, args)
//	stream.Synchronize()
//}
