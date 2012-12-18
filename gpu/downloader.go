package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
	"github.com/barnex/cuda5/cu"
)

// Downloads data from host to GPU.
type Downloader struct {
	host   nimble.Chan1
	dev    nimble.RChan1
	stream cu.Stream
}

func NewDownloader(tag, unit string, devdata_ nimble.Chan1) *Downloader {
	devdata := devdata_.NewReader()
	hostdata := nimble.MakeChan1(tag, unit, devdata.Mesh, nimble.CPUMemory, devdata_.NBufferedBlocks())
	MemHostRegister(hostdata.UnsafeData().Host())
	u := &Downloader{hostdata, devdata, cu.StreamCreate()}
	nimble.Stack(u)
	return u
}

func (u *Downloader) Output() nimble.Chan1 {
	return u.host
}

func (u *Downloader) Run() {
	core.Debug("run gpu.downloader")
	N := u.host.BlockLen()
	LockCudaThread()

	for {
		in := u.dev.ReadNext(N).Device()
		out := u.host.WriteNext(N).Host()
		in.CopyDtoHAsync(out, u.stream)
		u.stream.Synchronize()
		u.host.WriteDone()
		u.dev.ReadDone()
	}
}
