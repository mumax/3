package render

// Wrapper for glfw wrapper :-).
import (
	"github.com/jteeuwen/glfw"
)


func Init() {
}

func Close() {
	glfw.CloseWindow()
	glfw.Terminate()
}


func IsOpen() bool {
	return glfw.WindowParam(glfw.Opened) == 1
}

func VSync(vsync bool) {
	if vsync {
		glfw.SetSwapInterval(1)
	} else {
		glfw.SetSwapInterval(0)
	}
}

func SwapBuffers() {
	glfw.SwapBuffers()
}

func GrabMouse(grab bool) {
	if grab {
		glfw.Disable(glfw.MouseCursor)
	} else {
		glfw.Enable(glfw.MouseCursor)
	}
}
