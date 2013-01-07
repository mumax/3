package render

import (
	"code.google.com/p/mx3/dump"
	"code.google.com/p/mx3/gpu"
	"code.google.com/p/mx3/nimble"
	"github.com/jteeuwen/glfw"
	//"sync"
)

func Live(in_ nimble.ChanN) {
	go func() {
		in := in_.NewReader()

		Frame = new(dump.Frame)
		Frame.Components = in.NComp()
		Frame.MeshSize = in.Mesh().Size()
		Frame.MeshStep = in.Mesh().CellSize()
		Crop2 = Frame.MeshSize
		ncell := in.Mesh().NCell()
		n := ncell * in.NComp()
		Frame.Data = make([]float32, n)

		Init(800, 600, true, 2, true)
		InitInputHandlers()
		defer glfw.CloseWindow()
		defer glfw.Terminate()
		Viewpos[2] = -20

		//var lock sync.Mutex
		wantframe := true // TODO: atomic

		go func() {
			gpu.LockCudaThread()
			for {
				data := in.ReadNext(ncell)
				if wantframe {
					for i, d := range data {
						d.Device().CopyDtoH(Frame.Data[i*ncell : (i+1)*ncell])
					}
					wantframe = false
				}
				in.ReadDone()
			}
		}()

		for glfw.WindowParam(glfw.Opened) == 1 { // window open
			//lock.Lock()
			Render()
			if Wantscrot {
				Screenshot()
				Wantscrot = false
			}
			if wantframe == false {
				PreRender(Frame)
				wantframe = true
			}
			//lock.Unlock()
		}
	}()
}
