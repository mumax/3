package script

import (
	"log"
	"math"
	"reflect"
	"testing"
)

func init() {
	log.SetFlags(0)
}

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
	w.MustExec("x=3")
	if w.MustEval("x") != 3.0 {
		t.Fail()
	}
	w.MustExec("y:=8")
	if w.MustEval("y") != 8 {
		t.Error("got", w.MustEval("y"))
	}

	// Test Ops
	if w.MustEval("1+2*3/4-5-6") != 1.+2.*3./4.-5.-6 {
		t.Fail()
	}

	// Test func
	if w.MustEval("sqrt(3*3)").(float64) != 3 {
		t.Fail()
	}
}

func TestContains(t *testing.T) {
	w := NewWorld()

	var x float64
	w.Var("x", &x)

	X := w.Resolve("x")
	if X == nil {
		t.Fail()
	}

	if !Contains(w.MustCompile("x+1"), X) {
		t.Fail()
	}
	if Contains(w.MustCompile("1+1"), X) {
		t.Fail()
	}
}

func TestTypes(t *testing.T) {
	w := NewWorld()

	x := 3.14
	w.Var("x", &x)
	w.MustExec("x=7")

	w.Func("printInt", func(x int) { log.Println(x) })
	w.MustExec("printInt(7)")
}

func TestLoop(t *testing.T) {
	w := NewWorld()
	sum := 0.0
	w.Var("sum", &sum)
	src := `
		for i:=0; i<100; i++{
			sum = sum + i
		}
	`
	w.MustExec(src)
	if sum != 4950 {
		t.Error("got", sum)
	}

	src = `
		for i:=100; i>=0; i--{
			sum = sum + i
		}
	`
	w.MustExec(src)
	if sum != 10000 {
		t.Error("got", sum)
	}
}

type test struct {
	a, b, c int
}

func (t *test) A() int { return 41 }
func (t *test) B() int { return 42 }
func (t *test) C() int { return 43 }

func TestMethod(t *testing.T) {
	w := NewWorld()
	var s *test
	w.Var("s", &s)
	if w.MustEval("s.B()") != 42 {
		t.Fail()
	}
}

func TestScope(t *testing.T) {
	w := NewWorld()
	w.MustEval("sin(0)")
	w.EnterScope()
	w.MustEval("sin(0)")
	w.ExitScope()
	w.MustEval("sin(0)")
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

func BenchmarkEval2(b *testing.B) {
	b.StopTimer()
	w := NewWorld()
	code := w.MustCompileExpr("sin(cos(tan(log(sqrt(exp(1))))))")
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		code.Eval()
	}
}

func BenchmarkEval2_native(bench *testing.B) {
	var a float64
	b := 1.
	for i := 0; i < bench.N; i++ {
		a += math.Sin(math.Cos(math.Tan(math.Log(math.Sqrt(math.Exp(b))))))
	}
	if a == 1.23456 {
		panic("make sure result is used")
	}
}

type T struct {
	in  string
	out interface{}
}

func TestMany(test *testing.T) {
	tests := []T{
		{"1+1", 2.},
		{"7-5", 2.},
		{"2*3", 6.},
		{"10/5", 2.},
		{"1+10/5", 3.},
		{"10/5+1", 3.},
		{"(1+14)/5", 3.},
		{"1<1", false},
		{"1<2", true},
		{"2<1", false},
		{"1>1", false},
		{"2>1", true},
		{"1>2", false},
		{"1<=1", true},
		{"1<=2", true},
		{"2<=1", false},
		{"1>=1", true},
		{"2>=1", true},
		{"1>=2", false}}

	w := NewWorld()
	for _, t := range tests {
		out := w.MustEval(t.in)
		if !reflect.DeepEqual(out, t.out) {
			test.Error(t.in, "returned", out, "expected:", t.out)
		}
	}
}

// Test a few cases that should not compile
func TestFail(test *testing.T) {
	w := NewWorld()
	w.Const("c", 3e8)
	a := 1.
	w.Var("a", &a)
	tests := []string{"c=1", "undefined", "1++", "a=true", "x:=a++"}
	for _, t := range tests {
		_, err := w.Compile(t)
		if err == nil {
			test.Error(t, "should not compile")
		} else {
			log.Println(t, ":", err, ":OK")
		}
	}
}
