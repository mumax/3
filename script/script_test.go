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

	if w.EvalFloat64("1+2*3/4-5-6") != 1.+2.*3./4.-5.-6 {
		t.Fail()
	}
}
