//go:build darwin && arm64 && cgo

package metal

import (
	"fmt"
	"math"
	"sync"
	"testing"
	"unsafe"
)

const runtimeTestSource = `
#include <metal_stdlib>
using namespace metal;

kernel void mumax3_runtime_test_madd2(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    constant float &factor [[buffer(2)]],
    constant int &n [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < uint(n)) {
        dst[index] += factor * src[index];
    }
}

kernel void mumax3_runtime_test_optional_zero(
    device float *dst [[buffer(0)]],
    device const float *optional_value [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index < uint(n)) {
        dst[index] = optional_value[0];
    }
}
`

var (
	registerTestLibraryOnce sync.Once
	registerTestLibraryErr  error
)

func requireTestLibrary(tb testing.TB) {
	tb.Helper()
	registerTestLibraryOnce.Do(func() {
		registerTestLibraryErr = RegisterSource(runtimeTestSource)
	})
	if registerTestLibraryErr != nil {
		tb.Fatalf("register Metal test library: %v", registerTestLibraryErr)
	}
}

func TestAllocationCopyFillAndInteriorPointer(t *testing.T) {
	requireTestLibrary(t)

	const count = 257
	const offset = 17
	bytes := int64(count * 4)
	deviceDst := MustAlloc(bytes)
	deviceSrc := MustAlloc(bytes)
	defer func() {
		if err := Free(deviceSrc); err != nil {
			t.Errorf("free source: %v", err)
		}
		if err := Free(deviceDst); err != nil {
			t.Errorf("free destination: %v", err)
		}
	}()

	hostSrc := make([]float32, count)
	for index := range hostSrc {
		hostSrc[index] = float32(index) * 0.25
	}
	if err := CopyToDevice(
		deviceSrc,
		unsafe.Pointer(unsafe.SliceData(hostSrc)),
		bytes,
	); err != nil {
		t.Fatalf("host-to-device copy: %v", err)
	}
	if err := Fill(deviceDst, 0, bytes); err != nil {
		t.Fatalf("fill: %v", err)
	}
	base, allocationBytes, err := AllocationRange(
		unsafe.Add(deviceDst, offset*4),
	)
	if err != nil {
		t.Fatalf("query interior allocation: %v", err)
	}
	if base != deviceDst || allocationBytes != bytes {
		t.Fatalf(
			"allocation range = (%p, %d), want (%p, %d)",
			base,
			allocationBytes,
			deviceDst,
			bytes,
		)
	}

	dstInterior := unsafe.Add(deviceDst, offset*4)
	srcInterior := unsafe.Add(deviceSrc, offset*4)
	n := count - offset
	if err := Launch(
		"mumax3_runtime_test_madd2",
		Grid1D(n, 128),
		BufferArgN(dstInterior, uint64(n*4)),
		BufferArgN(srcInterior, uint64(n*4)),
		F32(2),
		I32(n),
	); err != nil {
		t.Fatalf("interior-pointer kernel launch: %v", err)
	}

	hostDst := make([]float32, count)
	if err := CopyToHost(
		unsafe.Pointer(unsafe.SliceData(hostDst)),
		deviceDst,
		bytes,
	); err != nil {
		t.Fatalf("device-to-host copy: %v", err)
	}
	for index := 0; index < offset; index++ {
		if hostDst[index] != 0 {
			t.Fatalf("prefix[%d] = %g, want zero", index, hostDst[index])
		}
	}
	for index := offset; index < count; index++ {
		want := 2 * hostSrc[index]
		if hostDst[index] != want {
			t.Fatalf("result[%d] = %g, want %g", index, hostDst[index], want)
		}
	}
}

func TestFillUint32(t *testing.T) {
	const count = 513
	const pattern = uint32(0x3fa00000) // float32(1.25)
	bytes := int64(count * 4)
	deviceBuffer := MustAlloc(bytes)
	defer func() {
		if err := Free(deviceBuffer); err != nil {
			t.Errorf("free buffer: %v", err)
		}
	}()
	if err := FillUint32(deviceBuffer, pattern, count); err != nil {
		t.Fatalf("uint32 fill: %v", err)
	}
	host := make([]uint32, count)
	if err := CopyToHost(
		unsafe.Pointer(unsafe.SliceData(host)),
		deviceBuffer,
		bytes,
	); err != nil {
		t.Fatalf("device-to-host copy: %v", err)
	}
	for index, value := range host {
		if value != pattern {
			t.Fatalf("result[%d] = %#x, want %#x", index, value, pattern)
		}
	}
}

func TestNilBufferUsesZeroAllocation(t *testing.T) {
	requireTestLibrary(t)

	const count = 33
	bytes := int64(count * 4)
	deviceDst := MustAlloc(bytes)
	defer func() {
		if err := Free(deviceDst); err != nil {
			t.Errorf("free destination: %v", err)
		}
	}()

	if err := Launch(
		"mumax3_runtime_test_optional_zero",
		Grid1D(count, 32),
		BufferArg(deviceDst),
		BufferArg(nil),
		I32(count),
	); err != nil {
		t.Fatalf("optional-buffer kernel launch: %v", err)
	}
	host := make([]float32, count)
	if err := CopyToHost(
		unsafe.Pointer(unsafe.SliceData(host)),
		deviceDst,
		bytes,
	); err != nil {
		t.Fatalf("device-to-host copy: %v", err)
	}
	for index, value := range host {
		if value != 0 {
			t.Fatalf("result[%d] = %g, want zero", index, value)
		}
	}
}

func TestBufferBoundsCheck(t *testing.T) {
	requireTestLibrary(t)

	deviceBuffer := MustAlloc(64)
	defer func() {
		if err := Free(deviceBuffer); err != nil {
			t.Errorf("free buffer: %v", err)
		}
	}()
	err := Launch(
		"mumax3_runtime_test_optional_zero",
		Grid1D(1, 1),
		BufferArgN(unsafe.Add(deviceBuffer, 60), 8),
		BufferArg(nil),
		I32(1),
	)
	if err == nil {
		t.Fatal("out-of-bounds interior buffer was accepted")
	}
}

func TestBatchedLaunchOrdering(t *testing.T) {
	requireTestLibrary(t)

	const count = 1024
	// Exceeds max_operations_per_command_buffer so this also verifies ordering
	// across the runtime's automatic command-buffer boundary.
	const launches = 300
	bytes := int64(count * 4)
	deviceDst := MustAlloc(bytes)
	deviceSrc := MustAlloc(bytes)
	defer func() {
		if err := Free(deviceSrc); err != nil {
			t.Errorf("free source: %v", err)
		}
		if err := Free(deviceDst); err != nil {
			t.Errorf("free destination: %v", err)
		}
	}()

	hostSrc := make([]float32, count)
	for index := range hostSrc {
		hostSrc[index] = 1
	}
	if err := CopyToDevice(
		deviceSrc,
		unsafe.Pointer(unsafe.SliceData(hostSrc)),
		bytes,
	); err != nil {
		t.Fatalf("host-to-device copy: %v", err)
	}
	if err := Fill(deviceDst, 0, bytes); err != nil {
		t.Fatalf("fill: %v", err)
	}
	for launch := 0; launch < launches; launch++ {
		if err := Launch(
			"mumax3_runtime_test_madd2",
			Grid1D(count, 256),
			BufferArg(deviceDst),
			BufferArg(deviceSrc),
			F32(1),
			I32(count),
		); err != nil {
			t.Fatalf("batch launch %d: %v", launch, err)
		}
	}
	if err := Sync(); err != nil {
		t.Fatalf("synchronize batch: %v", err)
	}

	hostDst := make([]float32, count)
	if err := CopyToHost(
		unsafe.Pointer(unsafe.SliceData(hostDst)),
		deviceDst,
		bytes,
	); err != nil {
		t.Fatalf("device-to-host copy: %v", err)
	}
	for index, value := range hostDst {
		if math.Abs(float64(value-launches)) != 0 {
			t.Fatalf("result[%d] = %g, want %d", index, value, launches)
		}
	}
}

func BenchmarkBatchedLaunchAndSync(b *testing.B) {
	requireTestLibrary(b)

	const count = 1 << 20
	const launchesPerBatch = 32
	bytes := int64(count * 4)
	deviceDst := MustAlloc(bytes)
	deviceSrc := MustAlloc(bytes)
	defer func() {
		if err := Free(deviceSrc); err != nil {
			b.Errorf("free source: %v", err)
		}
		if err := Free(deviceDst); err != nil {
			b.Errorf("free destination: %v", err)
		}
	}()
	if err := Fill(deviceDst, 0, bytes); err != nil {
		b.Fatalf("fill destination: %v", err)
	}
	if err := Fill(deviceSrc, 1, bytes); err != nil {
		b.Fatalf("fill source: %v", err)
	}
	if err := Sync(); err != nil {
		b.Fatalf("initial synchronize: %v", err)
	}

	cfg := Grid1D(count, 256)
	args := []Arg{
		BufferArg(deviceDst),
		BufferArg(deviceSrc),
		F32(1),
		I32(count),
	}
	b.SetBytes(bytes * 3 * launchesPerBatch)
	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		for launch := 0; launch < launchesPerBatch; launch++ {
			if err := Launch(
				"mumax3_runtime_test_madd2",
				cfg,
				args...,
			); err != nil {
				b.Fatal(err)
			}
		}
		if err := Sync(); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()

	info, err := Info()
	if err == nil {
		b.ReportMetric(
			float64(info.TrackedPeakAllocationSize)/(1024*1024),
			"peak-MiB",
		)
	}
}

func ExampleInfo() {
	info, err := Info()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%s, unified-memory=%t\n", info.Name, info.UnifiedMemory)
}
