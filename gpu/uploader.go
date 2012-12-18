package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
	"github.com/barnex/cuda5/cu"
)

// Uploads data from host to GPU.
type Uploader struct {
	host   nimble.RChan1
	dev    nimble.Chan1
	stream cu.Stream
}

func NewUploader(tag, unit string, hostdata_ nimble.Chan1) *Uploader {
	hostdata := hostdata_.NewReader()
	MemHostRegister(hostdata_.UnsafeData().Host())
	devdata := nimble.MakeChan1(tag, unit, hostdata.Mesh, nimble.GPUMemory, hostdata_.NBufferedBlocks())
	u := &Uploader{hostdata, devdata, cu.StreamCreate()}
	nimble.Stack(u)
	return u
}

func (u *Uploader) Output() nimble.Chan1 {
	return u.dev
}

func (u *Uploader) Run() {
	core.Debug("run gpu.uploader")
	N := u.host.BlockLen()
	LockCudaThread()

	for {
		in := u.host.ReadNext(N).Host()
		out := u.dev.WriteNext(N).Device()
		out.CopyHtoDAsync(in, u.stream)
		u.stream.Synchronize()
		u.dev.WriteDone()
		u.host.ReadDone()
	}
}
