// +build ignore

package main

import (
	"code.google.com/p/nimble-cube/core"
	. "code.google.com/p/nimble-cube/render"
	"flag"
	gl "github.com/chsc/gogl/gl21"
	"github.com/jteeuwen/glfw"
	"log"
	"time"
)

var (
	flag_smooth      = flag.Bool("smooth", false, "Smooth shading")
	flag_vsync       = flag.Bool("vsync", true, "Vertical sync")
	flag_cullface    = flag.Bool("cullface", true, "Cull invisible polygon faces")
	flag_lighting    = flag.Bool("lighting", false, "Enable lighting")
	flag_depthtest   = flag.Bool("depthtest", true, "Enable depth test")
	flag_antialias   = flag.Bool("antialias", false, "Antialias lines")
	flag_wireframe   = flag.Bool("wireframe", false, "Render wireframes")
	flag_fps         = flag.Bool("fps", true, "Measure frames per second")
	flag_multisample = flag.Int("multisample", 0, "Multisample")
)

func main() {
	flag.Parse()

	data := Load(flag.Arg(0))
	log.Println("data loaded")

	InitWindow()
	defer glfw.CloseWindow()
	defer glfw.Terminate()

	InitGL()
	InitViewport()
	InitInputHandlers()

	start := time.Now()
	frames := 0

	Viewpos[2] = -20

	for glfw.WindowParam(glfw.Opened) == 1 { // window open
		UpdateViewpos()
		//DrawTestScene()
		Render(data)
		glfw.SwapBuffers()
		frames++
		//glfw.WaitEvents()
	}

	if *flag_fps {
		fps := int((float64(frames) / float64(time.Since(start))) * float64(time.Second))
		log.Println("average FPS:", fps)
	}
}

// number of bits in buffer
const (
	r, g, b, a = 8, 8, 8, 8 // 0 means auto
	depth      = 16         // 0 means none
	stencil    = 0          // 0 means none
)

func InitWindow() {
	core.Fatal(glfw.Init())
	if *flag_multisample != 0 {
		glfw.OpenWindowHint(glfw.FsaaSamples, *flag_multisample)
	}
	Width, Height = 800, 600
	core.Fatal(glfw.OpenWindow(Width, Height, r, g, b, a, depth, stencil, glfw.Windowed))
	glfw.SetWindowTitle("renderer")
	vsync := 0
	if *flag_vsync {
		vsync = 1
	}
	glfw.SetSwapInterval(vsync)
}

func InitGL() {
	core.Fatal(gl.Init())

	if *flag_wireframe {
		gl.PolygonMode(gl.FRONT_AND_BACK, gl.LINE)
	}

	if *flag_depthtest {
		gl.Enable(gl.DEPTH_TEST)
		gl.DepthFunc(gl.LESS)
	}

	if *flag_lighting {
		gl.Enable(gl.LIGHTING)
	}

	if *flag_cullface {
		gl.Enable(gl.CULL_FACE)
		gl.CullFace(gl.BACK)
	}

	if *flag_antialias {
		gl.Enable(gl.LINE_SMOOTH)
		gl.Enable(gl.BLEND)
		gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
		gl.Hint(gl.LINE_SMOOTH_HINT, gl.NICEST)
	}

	if *flag_multisample != 0 {
		gl.Enable(gl.MULTISAMPLE)
	}

	if *flag_smooth {
		gl.ShadeModel(gl.SMOOTH)
	}
}
