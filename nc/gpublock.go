package nc

import (
	"github.com/barnex/cuda4/cu"
	"github.com/barnex/cuda4/safe"
	"math"
	"unsafe"
)

// Block of float32's on the GPU.
// TODO: could embed gpufloats
type GpuBlock struct {
	safe.Float32s
	size     [3]int
	refcount *Refcount
}

func MakeGpuBlock(size [3]int) GpuBlock {
	N := size[0] * size[1] * size[2]
	SetCudaCtx()
	return GpuBlock{safe.MakeFloat32s(N), size, nil}
}

func Make3GpuBlock(size [3]int) [3]GpuBlock {
	return [3]GpuBlock{MakeGpuBlock(size), MakeGpuBlock(size), MakeGpuBlock(size)}
}

func (g *GpuBlock) Free() {
	g.Float32s.Free()
	g.size = [3]int{0, 0, 0}
	g.refcount = nil
}

// Number of bytes for underlying storage.
func (b *GpuBlock) Bytes() int64 {
	return SIZEOF_FLOAT32 * int64(b.Len())
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
	SetCudaCtx()
	cu.MemcpyDtoH(dst.UnsafePointer(), src.Pointer(), src.Bytes())
}

// Copy from host to device.
func (dst *GpuBlock) CopyHtoD(src Block) {
	if src.Size() != dst.Size() {
		Panic("size mismatch:", src.Size(), dst.Size())
	}
	SetCudaCtx()
	cu.MemcpyHtoD(dst.Pointer(), src.UnsafePointer(), src.Bytes())
}

// Copy from device to host.
func (src *GpuBlock) CopyDtoHAsync(dst Block, str cu.Stream) {
	if src.Size() != dst.Size() {
		Panic("size mismatch:", src.Size(), dst.Size())
	}
	SetCudaCtx()
	cu.MemcpyDtoHAsync(dst.UnsafePointer(), src.Pointer(), src.Bytes(), str)
}

// Copy from host to device.
func (dst *GpuBlock) CopyHtoDAsync(src Block, str cu.Stream) {
	if src.Size() != dst.Size() {
		Panic("size mismatch:", src.Size(), dst.Size())
	}
	SetCudaCtx()
	cu.MemcpyHtoDAsync(dst.Pointer(), src.UnsafePointer(), src.Bytes(), str)
}

// Set all values to v.
func (b *GpuBlock) Memset(v float32) {
	SetCudaCtx()
	cu.MemsetD32(b.Pointer(), math.Float32bits(v), int64(b.Len()))
}

// Set the value at index i,j,k to v.
// Intended for debugging, prohibitively slow.
func (b *GpuBlock) Set(i, j, k int, v float32) {
	size := b.size
	V := v // addressable
	if i < 0 || j < 0 || k < 0 ||
		i >= size[0] || j >= size[1] || k >= size[2] {
		Panic("index out of bounds:", i, j, k, "size:", size)
	}
	I := i*size[1]*size[2] + j*size[2] + k
	offsetDst := cu.DevicePtr(uintptr(b.Pointer()) + SIZEOF_FLOAT32*uintptr(I))
	SetCudaCtx()
	cu.MemcpyHtoD(offsetDst, unsafe.Pointer(&V), SIZEOF_FLOAT32)
}

// Get the value at index i,j,k to v.
// Intended for debugging, prohibitively slow.
func (b *GpuBlock) Get(i, j, k int) float32 {
	size := b.size
	var V float32 // addressable
	if i < 0 || j < 0 || k < 0 ||
		i >= size[0] || j >= size[1] || k >= size[2] {
		Panic("index out of bounds:", i, j, k, "size:", size)
	}
	I := i*size[1]*size[2] + j*size[2] + k
	offsetSrc := cu.DevicePtr(uintptr(b.Pointer()) + SIZEOF_FLOAT32*uintptr(I))
	SetCudaCtx()
	cu.MemcpyDtoH(unsafe.Pointer(&V), offsetSrc, SIZEOF_FLOAT32)
	return V
}
