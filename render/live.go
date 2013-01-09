package render

import (
	"code.google.com/p/mx3/dump"
	"code.google.com/p/mx3/gpu"
	"code.google.com/p/mx3/nimble"
	"github.com/jteeuwen/glfw"
	"sync/atomic"
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

		wantframe := new(atomicbool)
		wantframe.set(true)

		go func() {
			gpu.LockCudaThread()
			for {
				data := in.ReadNext(ncell)
				if wantframe.get() {
					for i, d := range data {
						d.Device().CopyDtoH(Frame.Data[i*ncell : (i+1)*ncell])
					}
					wantframe.set(false)
				}
				in.ReadDone()
			}
		}()

		for glfw.WindowParam(glfw.Opened) == 1 { // window open
			Render()
			if Wantscrot {
				Screenshot()
				Wantscrot = false
			}
			if wantframe.get() == false {
				PreRender(Frame)
				wantframe.set(true)
			}
		}
	}()
}

type atomicbool int32

func (a *atomicbool) get() bool {
	return atomic.LoadInt32((*int32)(a)) == 1
}

func (a *atomicbool) set(v bool) {
	w := int32(0)
	if v {
		w = 1
	}
	atomic.StoreInt32((*int32)(a), w)
}
