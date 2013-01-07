package render

import (
	"code.google.com/p/mx3/core"
	gl "github.com/chsc/gogl/gl21"
	"github.com/jteeuwen/glfw"
	"log"
	"math"
)

// Parameters adjustable by keys
var (
	Viewpos      [3]int
	Rot          [3]int
	Crop1, Crop2 [3]int
	Light        [3]int
	Time         [3]int // only 1st element used.
	recording    bool   // screenshot at every key?
	//Ambient, Diffuse int
	//Frustum1, Frustum2 int
)
var N [3]int          // mesh size
var Width, Height int // window size

var (
	mousePrevX, mousePrevY int
	mouseButton            [5]int
)

var mode int

// variables targeted by mode keys
var keyTarget = map[int]*[3]int{
	P:   &Viewpos,
	C:   &Crop1,
	V:   &Crop2,
	R:   &Rot,
	T:   &Time,
	Esc: nil}

// called after updating the value of targeted by key
var postUpdate = map[int]func(){
	C: UpdateCrop,
	V: UpdateCrop,
	T: NextFrame}

// currently targeted variable
var activeTarget = &Viewpos

func InitKeyHandlers() {
	glfw.SetKeyCallback(func(key, state int) {
		if state == 0 {
			return
		}
		if key == Ret || key == Enter {
			Screenshot()
			return
		}
		if key == Q {
			recording = !recording
			log.Println("recording:", recording)
			return
		}

		if activeTarget != nil {
			switch key {
			case Left:
				(*activeTarget)[2]--
			case Right:
				(*activeTarget)[2]++
			case Down:
				(*activeTarget)[1]--
			case Up:
				(*activeTarget)[1]++
			case PgDown:
				(*activeTarget)[0]--
			case PgUp:
				(*activeTarget)[0]++
			}
		}

		if f, ok := postUpdate[mode]; ok {
			log.Println("post update", f)
			f()
		}

		if _, ok := keyTarget[key]; ok {
			activeTarget = keyTarget[key]
			mode = key
			log.Println("mode:", mode, "target:", activeTarget)
			return
		}

		if recording {
			Wantscrot = true
		}

		if core.DEBUG {
			log.Println("P:", Viewpos, "C:", Crop1, "V:", Crop2, "R:", Rot, "T:", Time[0])
		}
	})
}

func UpdateCrop() {
	LimitCrop()
	PreRender(Frame)
}

// Limit crop ranges to sensible values
func LimitCrop() {
	for i := range N {
		if Crop1[i] < 0 {
			Crop1[i] = 0
		}
		if Crop1[i] > N[i]-1 {
			Crop1[i] = N[i] - 1
		}
		if Crop2[i] < 1 {
			Crop2[i] = 1
		}
		if Crop2[i] > N[i] {
			Crop2[i] = N[i]
		}
		if Crop2[i] <= Crop1[i] {
			Crop2[i] = Crop1[i] + 1
		}
		if Crop1[i] >= Crop2[i] {
			Crop1[i] = Crop2[i] - 1
		}
	}
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
	gl.Rotatef(gl.Float(Rot[0]*2), 1, 0, 0)
	gl.Rotatef(gl.Float(Rot[1]*2), 0, 1, 0)
	gl.Rotatef(gl.Float(Rot[2]*2), 0, 0, 1)
}

const (
	deltaMove = 0.5
	deltaLook = 1
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

		Rot[0] += int(deltaLook * float64(dx))
		Rot[1] += int(deltaLook * float64(dy))

		// limit viewing angles
		for i := range Rot {
			if Rot[i] > 360 {
				Rot[i] = 0
			}
			if Rot[i] < 0 {
				Rot[i] += 360
			}
		}
		log.Println("rot:", Rot)
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
	Ret    = 294
	Enter  = 318
	Esc    = 257
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
	Q      = 81
)
