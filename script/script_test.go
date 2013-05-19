package script

import (
	"log"
	"testing"
)

func TestEval(t *testing.T) {
	log.SetFlags(0)

	w := NewWorld()

	// Test Variables
	x := 1.0
	w.Var("x", &x)
	if w.EvalFloat64("x") != 1.0 {
		t.Fail()
	}

	x = 2.0
	if w.EvalFloat64("x") != 2.0 {
		t.Fail()
	}

	y := 3.0
	w.Var("y", &y)
	if w.EvalFloat64("y") != y {
		t.Fail()
	}

}
