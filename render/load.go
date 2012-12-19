package render

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/dump"
)

var (
	framei int
	Frame  *dump.Frame
	pipe   <-chan *dump.Frame
)

func Load(fnames []string) {
	core.Log("loading", fnames)
	pipe = dump.ReadAllFiles(fnames, dump.CRC_ENABLED)
	Frame = <-pipe
	N = Frame.MeshSize
	Crop2 = Frame.MeshSize
	PreRender()
}

func NextFrame() {
	f := <-pipe
	if f == nil {
		core.Log("End of sequence")
		return
	}
	framei++
	core.Log("frame", framei)
	Frame = f
	N = Frame.MeshSize
	PreRender()
}
