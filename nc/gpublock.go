package nc

import (
	"github.com/barnex/cuda4/cu"
	"math"
)

// Block of float32's on the GPU.
// TODO: could embed gpufloats
type GpuBlock struct {
	ptr  cu.DevicePtr
	size [3]int
}

func MakeGpuBlock(size [3]int) GpuBlock {
	SetCudaCtx()
	N := size[0] * size[1] * size[2]
	return GpuBlock{cu.MemAlloc(cu.SIZEOF_FLOAT32 * int64(N)), size}
}

func Make3GpuBlock(size [3]int) [3]GpuBlock {
	return [3]GpuBlock{MakeGpuBlock(size), MakeGpuBlock(size), MakeGpuBlock(size)}
}

// Pointer to first element on the GPU.
func (b *GpuBlock) Pointer() cu.DevicePtr {
	return b.ptr
}

// Total number of scalar elements.
func (b *GpuBlock) N() int {
	return b.size[0] * b.size[1] * b.size[2]
}

// Number of bytes for underlying storage.
func (b *GpuBlock) Bytes() int64 {
	return SIZEOF_FLOAT32 * int64(b.N())
}

// BlockSize is the size of the block (N0, N1, N2)
// as was passed to MakeBlock()
func (b *GpuBlock) Size() [3]int {
	return b.size
}

// Make a copy on the host. Handy for debugging.
func (b *GpuBlock) Host() Block {
	host := MakeBlock(b.Size())
	b.CopyDtoH(host)
	return host
}

// Copy from device to host.
func (src *GpuBlock) CopyDtoH(dst Block) {
	if src.Size() != dst.Size() {
		Panic("size mismatch:", src.Size(), dst.Size())
	}
	cu.MemcpyDtoH(dst.UnsafePointer(), src.Pointer(), src.Bytes())
}

// Copy from host to device.
func (dst *GpuBlock) CopyHtoD(src Block) {
	if src.Size() != dst.Size() {
		Panic("size mismatch:", src.Size(), dst.Size())
	}
	cu.MemcpyHtoD(dst.Pointer(), src.UnsafePointer(), src.Bytes())
}

func (b *GpuBlock) Memset(v float32) {
	cu.MemsetD32(b.Pointer(), math.Float32bits(v), int64(b.N()))
}
