// +build ignore

package main

// test scripting language

import (
	"bytes"
	. "code.google.com/p/mx3/engine"
)

func main() {
	Init()
	defer Close()

	const script = `
		t
	`

	src := bytes.NewBuffer([]byte(script))
	RunScript(src)

}
