// +build ignore

package main

import (
	. "code.google.com/p/mx3/render"
	"github.com/jteeuwen/glfw"
	"os"
	"time"
)

const (
	W     = 800
	H     = 600
	Multi = 4
)

func main() {
	Load(os.Args[1:])

	Init(800, 600, true, Multi, false)
	defer glfw.CloseWindow()
	defer glfw.Terminate()

	Viewpos[2] = -20

	for {
		Render()
		time.Sleep(1 * time.Second)
		Screenshot()
		NextFrame()
	}

}
