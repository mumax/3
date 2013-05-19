package script

import (
	"log"
	"testing"
)

var tests = []string{"alpha=1 // cool"}

func TestEval(t *testing.T) {
	log.SetFlags(0)
	w := NewWorld()
	for _, str := range tests {
		c, err := w.Compile(str)
		if err != nil {
			log.Println(err)
		} else {
			log.Println(c)
		}
	}
}
