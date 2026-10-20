package metal

import (
	"math"
	"testing"
	"unsafe"
)

func TestArgumentEncoding(t *testing.T) {
	sentinel := byte(0)
	pointer := unsafe.Pointer(&sentinel)
	negative32 := int32(-42)
	negative64 := int64(-42)
	cases := []struct {
		name string
		arg  Arg
		kind argKind
		size uint32
		bits uint64
	}{
		{"buffer", BufferArg(pointer), argBuffer, 0, 0},
		{"float32", F32(1.25), argFloat32, 4, uint64(math.Float32bits(1.25))},
		{"int32", I32(-42), argInt32, 4, uint64(uint32(negative32))},
		{"uint32", U32(42), argUint32, 4, 42},
		{"uint8", U8(7), argUint8, 1, 7},
		{"float64", F64(1.25), argFloat64, 8, math.Float64bits(1.25)},
		{"int64", I64(-42), argInt64, 8, uint64(negative64)},
		{"uint64", U64(42), argUint64, 8, 42},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			if test.arg.kind != test.kind || test.arg.size != test.size || test.arg.bits != test.bits {
				t.Fatalf("unexpected encoding: %+v", test.arg)
			}
		})
	}
	if BufferArg(pointer).pointer != pointer {
		t.Fatal("buffer pointer was not preserved")
	}
}

func TestGrid1D(t *testing.T) {
	cfg := Grid1D(1025, 256)
	if cfg.GridX != 5 || cfg.GridY != 1 || cfg.GridZ != 1 {
		t.Fatalf("unexpected grid: %+v", cfg)
	}
	if cfg.BlockX != 256 || cfg.BlockY != 1 || cfg.BlockZ != 1 {
		t.Fatalf("unexpected block: %+v", cfg)
	}
}
