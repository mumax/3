package script

import (
	"log"
	"math"
	"testing"
)

func TestEval(t *testing.T) {
	log.SetFlags(0)

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

	if w.MustEval("1+2*3/4-5-6") != 1.+2.*3./4.-5.-6 {
		t.Fail()
	}

	w.Func("sqrt", math.Sqrt)
	if w.MustEval("sqrt(3*3)").(float64) != 3 {
		t.Fail()
	}

}
