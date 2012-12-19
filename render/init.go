package render

import (
	"code.google.com/p/mx3/core"
	gl "github.com/chsc/gogl/gl21"
	"github.com/jteeuwen/glfw"
)

// number of bits in buffer
const (
	r, g, b, a = 8, 8, 8, 8 // 0 means auto
	depth      = 16         // 0 means none
	stencil    = 0          // 0 means none
)

func Init(w, h int, smooth bool, multisample int, vsync bool) {
	InitWindow(800, 600, multisample, vsync)
	InitGL(smooth, multisample)
	InitViewport()
}

func InitWindow(w, h int, multisample int, vsync bool) {
	core.Fatal(glfw.Init())
	if multisample != 0 {
		glfw.OpenWindowHint(glfw.FsaaSamples, multisample)
	}
	Width, Height = w, h
	core.Fatal(glfw.OpenWindow(Width, Height, r, g, b, a, depth, stencil, glfw.Windowed))
	glfw.SetWindowTitle("renderer")
	if vsync {
		glfw.SetSwapInterval(1)
	}
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
