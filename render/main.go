// +build ignore

package main

import (
	"code.google.com/p/mx3/core"
	. "code.google.com/p/mx3/render"
	"flag"
	gl "github.com/chsc/gogl/gl21"
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

	InitWindow(800, 600)
	defer glfw.CloseWindow()
	defer glfw.Terminate()

	InitGL(*flag_smooth, *flag_multisample)
	InitViewport()
	InitInputHandlers()

	start := time.Now()
	frames := 0

	Viewpos[2] = -20

	for glfw.WindowParam(glfw.Opened) == 1 { // window open
		UpdateViewpos()
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

// number of bits in buffer
const (
	r, g, b, a = 8, 8, 8, 8 // 0 means auto
	depth      = 16         // 0 means none
	stencil    = 0          // 0 means none
)

func InitWindow(w, h int) {
	core.Fatal(glfw.Init())
	if *flag_multisample != 0 {
		glfw.OpenWindowHint(glfw.FsaaSamples, *flag_multisample)
	}
	Width, Height = w, h
	core.Fatal(glfw.OpenWindow(Width, Height, r, g, b, a, depth, stencil, glfw.Windowed))
	glfw.SetWindowTitle("renderer")
	glfw.SetSwapInterval(1)
}

func InitGL(smooth bool, multisample int) {
	core.Fatal(gl.Init())

	gl.Enable(gl.LIGHTING)

	gl.Enable(gl.CULL_FACE)
	gl.CullFace(gl.BACK)

	if multisample != 0 {
		gl.Enable(gl.MULTISAMPLE)
	}

	if smooth {
		gl.ShadeModel(gl.SMOOTH)
	}
}
