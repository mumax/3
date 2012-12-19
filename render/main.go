// +build ignore

package main

import (
	. "code.google.com/p/mx3/render"
	"flag"
	"github.com/jteeuwen/glfw"
	"log"
	"time"
)

var (
	flag_smooth      = flag.Bool("smooth", true, "Smooth shading")
	flag_multisample = flag.Int("multisample", 0, "Multisample")
)

func main() {
	flag.Parse()

	Load(flag.Args())

	Init(800, 600, *flag_smooth, *flag_multisample)
	defer glfw.CloseWindow()
	defer glfw.Terminate()

	start := time.Now()
	frames := 0

	Viewpos[2] = -20

	for glfw.WindowParam(glfw.Opened) == 1 { // window open
		Render()
		glfw.SwapBuffers()
		frames++
		if Wantscrot {
			Screenshot()
			Wantscrot = false
		}
		glfw.WaitEvents()
	}

	fps := int((float64(frames) / float64(time.Since(start))) * float64(time.Second))
	log.Println("average FPS:", fps)
}
