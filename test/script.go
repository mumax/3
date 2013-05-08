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
		msat = 800e3
		t = 1e-9
		Aex = 13e-12
		alpha = 1
	`
	src := bytes.NewBuffer([]byte(script))
	RunScript(src)

	check(Time, 1e-9)
	check(Aex(), 13e-12)
	check(Msat(), 12e-13)
	fmt.Println("OK")
}

func check(a, b float64) {
	if a != b {
		log.Fatal("expect ", b, " got ", a)
	}
}
