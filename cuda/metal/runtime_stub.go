//go:build !darwin || !arm64 || !cgo

package metal

import "unsafe"

func Available() bool {
	return false
}

func Initialize() error {
	return ErrUnsupported
}

func Close() error {
	return nil
}

func Info() (DeviceInfo, error) {
	return DeviceInfo{}, ErrUnsupported
}

func RegisterSource(source string) error {
	return ErrUnsupported
}

func RegisterLibrary(library []byte) error {
	return ErrUnsupported
}

func Alloc(bytes int64) (unsafe.Pointer, error) {
	return nil, ErrUnsupported
}

func MustAlloc(bytes int64) unsafe.Pointer {
	panic(ErrUnsupported)
}

func Free(pointer unsafe.Pointer) error {
	return ErrUnsupported
}

func AllocationRange(pointer unsafe.Pointer) (unsafe.Pointer, int64, error) {
	return nil, 0, ErrUnsupported
}

func Copy(dst, src unsafe.Pointer, bytes int64) error {
	return ErrUnsupported
}

func CopyToDevice(dst, src unsafe.Pointer, bytes int64) error {
	return ErrUnsupported
}

func CopyToHost(dst, src unsafe.Pointer, bytes int64) error {
	return ErrUnsupported
}

func Fill(dst unsafe.Pointer, value byte, bytes int64) error {
	return ErrUnsupported
}

func FillUint32(dst unsafe.Pointer, value uint32, count int64) error {
	return ErrUnsupported
}

func Launch(name string, cfg GridConfig, args ...Arg) error {
	return ErrUnsupported
}

func MustLaunch(name string, cfg GridConfig, args ...Arg) {
	panic(ErrUnsupported)
}

func Flush() error {
	return ErrUnsupported
}

func Sync() error {
	return ErrUnsupported
}
