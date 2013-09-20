package cuda

import (
	"github.com/mumax/3/data"
	"testing"
)

func init() {
	Init(0, "auto")
}

func TestSlice(t *testing.T) {
	N0, N1, N2 := 2, 4, 8
	c := 1e-6
	m := data.NewMesh(N0, N1, N2, c, c, c)
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
	if *b.Mesh() != *a.Mesh() {
		t.Fail()
	}
}

func TestCpy(t *testing.T) {
	N0, N1, N2 := 2, 4, 32
	N := N0 * N1 * N2
	mesh := data.NewMesh(N0, N1, N2, 1, 1, 1)

	h1 := make([]float32, N)
	for i := range h1 {
		h1[i] = float32(i)
	}
	hs := data.SliceFromList([][]float32{h1}, mesh)

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
	LockThread()
	N0, N1, N2 := 128, 1024, 1024
	c := 1e-6
	m := data.NewMesh(N0, N1, N2, c, c, c)
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
	LockThread()
	N0, N1, N2 := 1, 10, 10
	c := 1e-6
	m := data.NewMesh(N0, N1, N2, c, c, c)
	a := NewUnifiedSlice(3, m)
	defer a.Free()

	b := a.Host()
	if b[0][0] != 0 || b[1][42] != 0 || b[2][99] != 0 {
		t.Fail()
	}

	Memset(a, 1, 2, 3)
	b = a.Host()
	if b[0][0] != 1 || b[1][42] != 2 || b[2][99] != 3 {
		t.Fail()
	}
}

func TestSliceSlice(t *testing.T) {
	LockThread()
	N0, N1, N2 := 1, 10, 10
	c := 1e-6
	m := data.NewMesh(N0, N1, N2, c, c, c)
	a := NewUnifiedSlice(3, m)
	h := a.Host()
	h[1][21] = 42
	b := a.Slice(20, 30)
	if b.Len() != 30-20 {
		t.Fail()
	}
	if b.NComp() != a.NComp() {
		t.Fail()
	}
	if b.Host()[1][1] != 42 {
		t.Fail()
	}
	if *a.Mesh() != *b.Mesh() {
		t.Fail()
	}
	if a.MemType() != b.MemType() {
		t.Fail()
	}
}
