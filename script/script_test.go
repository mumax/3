package script

import (
	"log"
	"testing"
)

func TestEval(t *testing.T) {
	log.SetFlags(0)

	w := NewWorld()

	alpha := 0.5
	w.Var("alpha", &alpha)

	log.Println(w.MustEval("alpha"))

}
