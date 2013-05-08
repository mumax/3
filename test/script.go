// +build ignore

package main

// test scripting language

import (
	"bytes"
	. "code.google.com/p/mx3/engine"
	"fmt"
	"log"
)

func main() {
	Init()
	defer Close()

	const script = `
		t = 1e-9
		msat = 800e3
		aex = 13e-12
		b_ext = (alpha, 2e-3, 3e-3)
		alpha = t
		print(alpha)
		setmesh(8, 4, 1, 1e-9, 1e-9, 1e-9)
	`
	src := bytes.NewBuffer([]byte(script))
	RunScript(src)

	check(Time, 1e-9)
	check(Aex(), 13e-12)
	check(Msat(), 800e3)
	check(B_ext()[0], Alpha())
	check(B_ext()[1], 2e-3)
	check(B_ext()[2], 3e-3)
	check(Alpha(), Time)

	// check functional semantics
	Time = 77e-9
	check(B_ext()[0], Alpha())
	check(Alpha(), Time)

	fmt.Println("OK")
}

func check(a, b float64) {
	if a != b {
		log.Fatal("expect ", b, " got ", a)
	}
}
