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
	flag_smooth    = flag.Bool("smooth", true, "Smooth shading")
	flag_vsync     = flag.Bool("vsync", true, "Vertical sync")
	flag_cullface  = flag.Bool("cullface", true, "Cull invisible polygon faces")
	flag_lighting  = flag.Bool("lighting", true, "Enable lighting")
	flag_depthtest = flag.Bool("depthtest", true, "Enable depth test")
	flag_antialias = flag.Bool("antialias", true, "Antialias lines")
	flag_wireframe = flag.Bool("wireframe", false, "Render wireframes")
	flag_fps       = flag.Bool("fps", true, "Measure frames per second")
)

var Width, Height int

func main() {
	flag.Parse()

	InitWindow()
	defer glfw.CloseWindow()
	defer glfw.Terminate()

	InitInputHandlers()
	InitGL()
	InitViewport()

	start := time.Now()
	frames := 0

	Viewpos.Z = 10

	for glfw.WindowParam(glfw.Opened) == 1 { // window open
		UpdateViewpos()
		DrawTestScene()
		glfw.SwapBuffers()
		frames++
	}

	if *flag_fps {
		fps := int((float64(frames) / float64(time.Since(start))) * float64(time.Second))
		log.Println("average FPS:", fps)
	}
}

func InitWindow() {
	core.Fatal(glfw.Init())
	Width, Height = 800, 600
	core.Fatal(glfw.OpenWindow(Width, Height, 0, 0, 0, 0, 0, 0, glfw.Windowed))
	glfw.SetWindowTitle("renderer")
	vsync := 0
	if *flag_vsync {
		vsync = 1
	}
	glfw.SetSwapInterval(vsync)
}

func InitViewport() {
	gl.Viewport(0, 0, gl.Sizei(Width), gl.Sizei(Height))
	gl.MatrixMode(gl.PROJECTION)
	gl.LoadIdentity()
	x := gl.Double(float64(Height) / float64(Width))
	gl.Frustum(-1, 1, -x, x, 1, 1000.0)
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

	if *flag_smooth {
		gl.ShadeModel(gl.SMOOTH)
	}
}
