package safe

import (
	"reflect"
	"testing"
)

func TestComplex128sSlice(test *testing.T) {
	InitCuda()

	a := MakeComplex128s(100)
	defer a.Free()

	if !reflect.DeepEqual(a.Host(), make([]complex128, 100)) {
		test.Error(a.Host())
	}

	b := make([]complex128, 100)

	if a.Len() != len(b) {
		test.Error("len:", a.Len(), "!=", cap(b))
	}
	if a.Cap() != cap(b) {
		test.Error("cap:", a.Cap(), "!=", cap(b))
	}

	c := a.Slice(20, 30)
	d := b[20:30]

	if c.Len() != len(d) {
		test.Error("sliced len:", c.Len(), "!=", cap(d))
	}
	if c.Cap() != cap(d) {
		test.Error("sliced cap:", c.Cap(), "!=", cap(d))
	}

	e := a.Slice(0, 50)
	f := b[0:50]

	if e.Len() != len(f) {
		test.Error("sliced len:", e.Len(), "!=", cap(f))
	}
	if e.Cap() != cap(f) {
		test.Error("sliced cap:", e.Cap(), "!=", cap(f))
	}
}

func TestComplex128sPanic1(test *testing.T) {
	InitCuda()

	defer func() {
		err := recover()
		test.Log("recovered:", err)
		if err == nil {
			test.Fail()
		}
	}()

	a := MakeComplex128s(100)
	defer a.Free()

	a.Slice(-1, 10)
}

func TestComplex128sPanic2(test *testing.T) {
	InitCuda()

	defer func() {
		err := recover()
		test.Log("recovered:", err)
		if err == nil {
			test.Fail()
		}
	}()

	a := MakeComplex128s(100)
	defer a.Free()

	a.Slice(0, 101)
}

func TestComplex128sCopy(test *testing.T) {
	InitCuda()

	a := make([]complex128, 100)

	b := MakeComplex128s(100)
	defer b.Free()

	c := MakeComplex128s(100)
	defer c.Free()

	d := make([]complex128, 200)

	for i := range a {
		a[i] = complex(float64(i), float64(2*i))
	}

	b.CopyHtoD(a)

	c.CopyDtoD(b)

	c.CopyDtoH(d[:100])

	if !reflect.DeepEqual(a, d[:100]) {
		test.Error(d)
	}
	if !reflect.DeepEqual(d[100:], make([]complex128, 100)) {
		test.Error(d)
	}
}
