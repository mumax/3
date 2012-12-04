/*
 Wrapper for glfw wrapper :-).
 Allows to swap-in other libs.
*/
package render

import (
	"github.com/jteeuwen/glfw"
)

import (
	"log"
)

func Version() string {
	return "glfw"
}

func Init() {
	if err := glfw.Init(); err != nil {
		log.Fatalf("glfw: %s\n", err)
		return
	}
}

func Close() {
	glfw.CloseWindow()
	glfw.Terminate()
}

func Windowed(w, h int, title string) {
	if err := glfw.OpenWindow(w, h, 0, 0, 0, 0, 0, 0, glfw.Windowed); err != nil {
		log.Fatalf("glfw: %s\n", err)
	}
	glfw.SetWindowTitle(title)
}

func Fullscreen(title string) (width, height int) {
	desktop := glfw.DesktopMode()
	width = desktop.W
	height = desktop.H
	if err := glfw.OpenWindow(width, height, 0, 0, 0, 0, 0, 0, glfw.Fullscreen); err != nil {
		log.Fatalf("glfw: %s\n", err)
	}
	glfw.SetWindowTitle(title)
	return
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
