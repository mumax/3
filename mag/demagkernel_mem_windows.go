package mag

import (
	"syscall"
	"unsafe"
)

// availableMemory returns the available physical memory in bytes (GlobalMemoryStatusEx), 0 if unknown.
// Used to size CalcDemagKernelFast's image cache.
func availableMemory() int64 {
	var m struct { // MEMORYSTATUSEX
		length, memoryLoad                          uint32
		totalPhys, availPhys                        uint64
		totalPageFile, availPageFile                uint64
		totalVirtual, availVirtual, availExtVirtual uint64
	}
	m.length = uint32(unsafe.Sizeof(m))
	ok, _, _ := syscall.NewLazyDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx").Call(uintptr(unsafe.Pointer(&m)))
	if ok == 0 {
		return 0
	}
	return int64(m.availPhys)
}
