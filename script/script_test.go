package script

import (
	"math"
	"testing"
)

func TestEval(t *testing.T) {
	w := NewWorld()

	// Test Variables
	x := 1.0
	w.Var("x", &x)
	if w.MustEval("x") != 1.0 {
		t.Fail()
	}
	x = 2.0
	if w.MustEval("x") != 2.0 {
		t.Fail()
	}

	// Test Ops
	if w.MustEval("1+2*3/4-5-6") != 1.+2.*3./4.-5.-6 {
		t.Fail()
	}

	// Test func
	w.Func("sqrt", math.Sqrt)
	if w.MustEval("sqrt(3*3)").(float64) != 3 {
		t.Fail()
	}
}

func BenchmarkEval1(b *testing.B) {
	b.StopTimer()
	w := NewWorld()
	code := w.MustCompileExpr("1+(2-3)*(4+5)/6")
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		code.Eval()
	}
}

func BenchmarkEval1_native(bench *testing.B) {
	var a, b, c, d, e, f float64
	for i := 0; i < bench.N; i++ {
		a += (b - c) * (d + e) / f
	}
	if a == 1 {
		panic("make sure result is used")
	}
}
