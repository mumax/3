package safe

import (
	"reflect"
	"testing"
)

func TestFloat64sSlice(test *testing.T) {
	InitCuda()

	a := MakeFloat64s(100)
	defer a.Free()

	if !reflect.DeepEqual(a.Host(), make([]float64, 100)) {
		test.Error(a.Host())
	}

	b := make([]float64, 100)

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

func TestFloat64sPanic1(test *testing.T) {
	InitCuda()

	defer func() {
		err := recover()
		test.Log("recovered:", err)
		if err == nil {
			test.Fail()
		}
	}()

	a := MakeFloat64s(100)
	defer a.Free()

	a.Slice(-1, 10)
}

func TestFloat64sPanic2(test *testing.T) {
	InitCuda()

	defer func() {
		err := recover()
		test.Log("recovered:", err)
		if err == nil {
			test.Fail()
		}
	}()

	a := MakeFloat64s(100)
	defer a.Free()

	a.Slice(0, 101)
}

func TestFloat64sCopy(test *testing.T) {
	InitCuda()

	a := make([]float64, 100)

	b := MakeFloat64s(100)
	defer b.Free()

	c := MakeFloat64s(100)
	defer c.Free()

	d := make([]float64, 200)

	for i := range a {
		a[i] = float64(i)
	}

	b.CopyHtoD(a)

	c.CopyDtoD(b)

	c.CopyDtoH(d[:100])

	if !reflect.DeepEqual(a, d[:100]) {
		test.Error(d)
	}
	if !reflect.DeepEqual(d[100:], make([]float64, 100)) {
		test.Error(d)
	}
}
