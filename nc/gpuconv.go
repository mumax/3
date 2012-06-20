package nc

import (
	"github.com/barnex/cuda4/cu"
	"unsafe"
)

type GpuConvBox struct {
	M      [3]<-chan GpuBlock
	B      [][3]chan<- GpuBlock
	Kernel [3][3]<-chan GpuBlock
	fftBuf [3]GpuBlock
}

func NewGpuConvBox() *GpuConvBox {
	box := new(GpuConvBox)
	Register(box)
	return box
}

func (box *GpuConvBox) Run() {

	size := Size()

	padded := [3]int{
		size[0] * 2,
		size[1] * 2,
		size[2] * 2}

	fftSize := [3]int{
		padded[0],
		padded[1],
		padded[2] + 2}

	box.fftBuf = Make3GpuBlock(fftSize)
	fftBuf := box.fftBuf

	for {
		for c := 0; c < 3; c++ {
			fftBuf[c].Memset(0)
			for s := 0; s < NumWarp(); s++ {
				m := RecvGpu(box.M[c])
				copyPad(fftBuf[c], m, sliceOffset(s))
			}
			Debug("fftbuf:", fftBuf[c].Host())
		}
	}
}

func sliceOffset(s int) [3]int {
	return [3]int{0, 0, 0} // TODO
}

var (
	copyPadKern cu.Function
)

func PtxDir() string {
	return "/home/arne/go/src/nimble-cube/ptx"
	//return ExecutableDir() + "../src/nimble-cube/ptx/" // KO with go run
}

func copyPad(dst GpuBlock, src GpuBlock, offset [3]int) {

	if copyPadKern == 0 {
		ptx := PtxDir() + "/copypad.ptx"
		Debug("Loading", ptx)
		mod := cu.ModuleLoad(ptx)
		copyPadKern = mod.GetFunction("copypad")
	}

	dstptr := dst.Pointer()
	dstsize := dst.Size()
	srcptr := src.Pointer()
	srcsize := src.Size()

	block := 16
	gridJ := DivUp(srcsize[1], block)
	gridK := DivUp(srcsize[2], block)
	shmem := 0
	args := []unsafe.Pointer{
		unsafe.Pointer(&dstptr),
		unsafe.Pointer(&dstsize[0]),
		unsafe.Pointer(&dstsize[1]),
		unsafe.Pointer(&dstsize[2]),
		unsafe.Pointer(&srcptr),
		unsafe.Pointer(&srcsize[0]),
		unsafe.Pointer(&srcsize[1]),
		unsafe.Pointer(&srcsize[2]),
		unsafe.Pointer(&offset[0]),
		unsafe.Pointer(&offset[1]),
		unsafe.Pointer(&offset[2])}

	cu.LaunchKernel(copyPadKern, gridJ, gridK, 1, block, block, 1, shmem, 0, args)
}

// Integer division rounded up.
func DivUp(x, y int) int {
	return ((x - 1) / y) + 1
}
