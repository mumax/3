package render

import (
	gl "github.com/chsc/gogl/gl21"
	"github.com/jteeuwen/glfw"
	//"log"
	"math"
)

// Adjustable parameters
var (
	Viewpos            [3]float32
	ViewPhi, ViewTheta float64
	Crop1, Crop2       [3]int
	Light              [3]float32
	//Ambient, Diffuse float32
	//Frustrum1, Frustrum2 int
)

var (
	mousePrevX, mousePrevY int
	mouseButton            [5]int
)

var Width, Height int

func InitKeyHandlers() {
	glfw.SetKeyCallback(func(key, state int) {
		if state == 1 {

		}
	})
}

const PI = math.Pi

func InitViewport() {
	gl.Viewport(0, 0, gl.Sizei(Width), gl.Sizei(Height))
	gl.MatrixMode(gl.PROJECTION)
	gl.LoadIdentity()
	x := gl.Double(float64(Height) / float64(Width))
	gl.Frustum(-1./2., 1./2., -x/2, x/2, 10, 100)
}

// Set the GL modelview matrix to match view position and orientation.
func UpdateViewpos() {
	gl.MatrixMode(gl.MODELVIEW)
	gl.LoadIdentity()
	gl.Translatef(gl.Float(Viewpos[0]), gl.Float(Viewpos[1]), gl.Float(Viewpos[2]))
	gl.Rotatef(gl.Float(ViewTheta*(180/PI)), 1, 0, 0)
	gl.Rotatef(gl.Float(ViewPhi*(180/PI)), 0, 0, 1)
}

const (
	deltaMove = 0.5
	deltaLook = 0.01
)

// Sets up input handlers
func InitInputHandlers() {
	InitKeyHandlers()

	glfw.SetMousePosCallback(func(x, y int) {

		dx, dy := x-mousePrevX, y-mousePrevY
		mousePrevX, mousePrevY = x, y
		if mouseButton[0] == 0 {
			return
		}

		ViewPhi += deltaLook * float64(dx)
		ViewTheta += deltaLook * float64(dy)

		// limit viewing angles
		if ViewPhi < -PI {
			ViewPhi += 2 * PI
		}
		if ViewPhi > PI {
			ViewPhi -= 2 * PI
		}
		if ViewTheta > PI {
			ViewTheta = -PI
		}
		if ViewTheta < -PI {
			ViewTheta = PI
		}
		//log.Println("phi, theta:", ViewPhi, ViewTheta)
	})

	glfw.SetMouseButtonCallback(func(button, state int) {
		//log.Println("mousebutton:", button, state)
		mouseButton[button] = state
	})

	glfw.SetMouseWheelCallback(func(delta int) {
		//log.Println("mousewheel:", delta)
		glfw.SetMouseWheel(0)
	})

	glfw.SetWindowSizeCallback(func(w, h int) {
		Width, Height = w, h
		InitViewport()
	})
}

const (
	Up     = 283
	Down   = 284
	Left   = 285
	Right  = 286
	PgUp   = 298
	PgDown = 299
	Esc    = 257
	Ret    = 294
	Enter  = 318
	P      = 80
	C      = 67
	V      = 86
	R      = 82
	F      = 70
	G      = 71
	T      = 84
	L      = 76
	A      = 65
	D      = 68
)
