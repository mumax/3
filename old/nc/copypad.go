package nc

import (
	"github.com/barnex/cuda4/cu"
	"unsafe"
)

var copyPadKern cu.Function

// Copies src into dst (which is larger), at offset position.
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
