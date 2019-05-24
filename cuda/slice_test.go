package cuda

import (
	"testing"

	"github.com/mumax/3/data"
)

func TestSlice(t *testing.T) {
	N0, N1, N2 := 2, 4, 8
	m := [3]int{N0, N1, N2}
	N := N0 * N1 * N2

	a := NewSlice(3, m)
	defer a.Free()
	Memset(a, 1, 2, 3)

	if a.GPUAccess() == false {
		t.Fail()
	}
	if a.Len() != N {
		t.Fail()
	}
	if a.NComp() != 3 {
		t.Fail()
	}

	b := a.Comp(1)
	if b.GPUAccess() == false {
		t.Error("b.GPUAccess", b.GPUAccess())
	}
	if b.Len() != N {
		t.Error("b.Len", b.Len())
	}
	if b.NComp() != 1 {
		t.Error("b.NComp", b.NComp())
	}
	if b.Size() != a.Size() {
		t.Fail()
	}
}

func TestCpy(t *testing.T) {
	N0, N1, N2 := 2, 4, 32
	N := N0 * N1 * N2
	mesh := [3]int{N0, N1, N2}

	h1 := make([]float32, N)
	for i := range h1 {
		h1[i] = float32(i)
	}
	hs := sliceFromList([][]float32{h1}, mesh)

	d := NewSlice(1, mesh)
	data.Copy(d, hs)

	d2 := NewSlice(1, mesh)
	data.Copy(d2, d)

	h2 := data.NewSlice(1, mesh)
	data.Copy(h2, d2)

	res := h2.Host()[0]
	for i := range res {
		if res[i] != h1[i] {
			t.Fail()
		}
	}
}

func TestSliceFree(t *testing.T) {
	N0, N1, N2 := 128, 1024, 1024
	m := [3]int{N0, N1, N2}
	N := 17
	// not freeing would attempt to allocate 17GB.
	for i := 0; i < N; i++ {
		a := NewSlice(2, m)
		a.Free()
	}
	a := NewSlice(2, m)
	a.Free()
	a.Free() // test double-free
}

func TestSliceHost(t *testing.T) {
	N0, N1, N2 := 1, 10, 10
	m := [3]int{N0, N1, N2}
	a := NewSlice(3, m)
	defer a.Free()

	b := a.HostCopy().Host()
	if b[0][0] != 0 || b[1][42] != 0 || b[2][99] != 0 {
		t.Error("slice not inited to zero")
	}

	Memset(a, 1, 2, 3)
	b = a.HostCopy().Host()
	if b[0][0] != 1 || b[1][42] != 2 || b[2][99] != 3 {
		t.Error("slice memset")
	}
}
