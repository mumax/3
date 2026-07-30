package metal

import (
	"errors"
	"fmt"
	"math"
	"unsafe"
)

// ErrUnsupported is returned when the Metal runtime is used on a platform
// other than cgo-enabled macOS on Apple Silicon.
var ErrUnsupported = errors.New("mumax3: Metal backend requires macOS on Apple Silicon with cgo enabled")

const (
	// Backend is the stable user-facing name of this compute backend.
	Backend = "Metal"

	// BackendVersion is the Metal feature baseline used by the native kernels.
	BackendVersion = "3"
)

// GridConfig describes a CUDA-compatible launch geometry. Grid* is the number
// of threadgroups and Block* is the number of threads in each threadgroup.
// Metal receives Grid*Block threads in each dimension.
type GridConfig struct {
	GridX  uint32
	GridY  uint32
	GridZ  uint32
	BlockX uint32
	BlockY uint32
	BlockZ uint32
}

// Grid returns a launch geometry with explicit three-dimensional dimensions.
func Grid(gridX, gridY, gridZ, blockX, blockY, blockZ int) GridConfig {
	return GridConfig{
		GridX:  checkedDimension("grid x", gridX),
		GridY:  checkedDimension("grid y", gridY),
		GridZ:  checkedDimension("grid z", gridZ),
		BlockX: checkedDimension("block x", blockX),
		BlockY: checkedDimension("block y", blockY),
		BlockZ: checkedDimension("block z", blockZ),
	}
}

// Grid1D returns the launch geometry needed to process n independent items.
func Grid1D(n, blockSize int) GridConfig {
	if n < 0 {
		panic("mumax3/metal: negative item count")
	}
	if blockSize <= 0 {
		panic("mumax3/metal: block size must be positive")
	}
	blocks := (uint64(n) + uint64(blockSize) - 1) / uint64(blockSize)
	if blocks > math.MaxUint32 {
		panic("mumax3/metal: launch grid exceeds uint32")
	}
	return GridConfig{
		GridX:  uint32(blocks),
		GridY:  1,
		GridZ:  1,
		BlockX: checkedDimension("block x", blockSize),
		BlockY: 1,
		BlockZ: 1,
	}
}

func checkedDimension(name string, value int) uint32 {
	if value < 0 || uint64(value) > math.MaxUint32 {
		panic(fmt.Sprintf("mumax3/metal: invalid %s dimension %d", name, value))
	}
	return uint32(value)
}

type argKind uint32

const (
	argBuffer argKind = iota + 1
	argFloat32
	argInt32
	argUint32
	argUint8
	argFloat64
	argInt64
	argUint64
)

// Arg is one Metal kernel argument. Values are deliberately opaque so callers
// cannot construct an argument whose size disagrees with its kind.
type Arg struct {
	kind    argKind
	size    uint32
	pointer unsafe.Pointer
	bits    uint64
}

// BufferArg binds an allocation returned by Alloc. Interior addresses are
// valid: the runtime resolves them to the owning MTLBuffer and byte offset.
// A nil pointer binds a small read-only zero buffer for optional kernel inputs.
func BufferArg(pointer unsafe.Pointer) Arg {
	return Arg{kind: argBuffer, pointer: pointer}
}

// BufferArgN is BufferArg with an additional bounds check for at least bytes
// accessible from pointer.
func BufferArgN(pointer unsafe.Pointer, bytes uint64) Arg {
	return Arg{kind: argBuffer, size: checkedSize(bytes), pointer: pointer}
}

// F32 binds a 32-bit floating-point scalar.
func F32(value float32) Arg {
	return Arg{kind: argFloat32, size: 4, bits: uint64(math.Float32bits(value))}
}

// I32 binds a 32-bit signed scalar. It panics rather than silently truncating.
func I32(value int) Arg {
	if int64(value) < math.MinInt32 || int64(value) > math.MaxInt32 {
		panic(fmt.Sprintf("mumax3/metal: int32 argument out of range: %d", value))
	}
	return Arg{kind: argInt32, size: 4, bits: uint64(uint32(int32(value)))}
}

// U32 binds a 32-bit unsigned scalar.
func U32(value uint32) Arg {
	return Arg{kind: argUint32, size: 4, bits: uint64(value)}
}

// U8 binds an 8-bit unsigned scalar.
func U8(value byte) Arg {
	return Arg{kind: argUint8, size: 1, bits: uint64(value)}
}

// F64 binds a 64-bit floating-point scalar. MuMax3's current GPU kernels use
// F32; F64 is provided for host-side adapters and future hardware.
func F64(value float64) Arg {
	return Arg{kind: argFloat64, size: 8, bits: math.Float64bits(value)}
}

// I64 binds a 64-bit signed scalar.
func I64(value int64) Arg {
	return Arg{kind: argInt64, size: 8, bits: uint64(value)}
}

// U64 binds a 64-bit unsigned scalar.
func U64(value uint64) Arg {
	return Arg{kind: argUint64, size: 8, bits: value}
}

func checkedSize(bytes uint64) uint32 {
	if bytes > math.MaxUint32 {
		panic(fmt.Sprintf("mumax3/metal: checked buffer span is too large: %d", bytes))
	}
	return uint32(bytes)
}

// DeviceInfo describes the selected Metal device and current runtime limits.
type DeviceInfo struct {
	Name                         string
	UnifiedMemory                bool
	MaxThreadsPerThreadgroup     [3]uint64
	RecommendedMaxWorkingSetSize uint64
	CurrentAllocatedSize         uint64
	TrackedAllocationSize        uint64
	TrackedPeakAllocationSize    uint64
}

// RuntimeError carries the stable C ABI error code as well as its diagnostic.
type RuntimeError struct {
	Code    int
	Message string
}

func (err *RuntimeError) Error() string {
	return fmt.Sprintf("mumax3/metal: %s (runtime code %d)", err.Message, err.Code)
}
