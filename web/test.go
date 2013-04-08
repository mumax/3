// +build ignore

package main

import (
	"code.google.com/p/mx3/engine"
	"code.google.com/p/mx3/web"
)

func main() {
	engine.Init()
	defer engine.Close()

	web.Serve(":8080")
}
