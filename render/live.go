// +build ignore

package main

import (
	"code.google.com/p/mx3/frame"
	"code.google.com/p/mx3/nimble"
	. "code.google.com/p/mx3/render"
	"flag"
	"github.com/jteeuwen/glfw"
	"log"
	"time"
)

func Live(in_ nimble.ChanN) {
	//	in := in_.NewReader()
	//
	//	Frame = new(dump.Frame)
	//	Frame.Components = in.NComp()
	//	Frame.MeshSize = in.Mesh().Size()
	//	Frame.MeshStep = in.Mesh().CellSize()
	//	n := in.Mesh().NCell() * in.NComp()
	//	Frame.Data = make([]float32, n)
	//
	//	Init(800, 600, *flag_smooth, *flag_multisample, true)
	//	InitInputHandlers()
	//	defer glfw.CloseWindow()
	//	defer glfw.Terminate()
	//
	//	Viewpos[2] = -20
	//
	//	for glfw.WindowParam(glfw.Opened) == 1 { // window open
	// 
	//		if data, ok := in.TryRead(n); ok{
	//
	//		}
	//
	//		Render()
	//		if Wantscrot {
	//			Screenshot()
	//			Wantscrot = false
	//		}
	//		wantframe <- 1
	//
	//	}

}
