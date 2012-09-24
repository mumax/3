package gpu

import (
	"github.com/barnex/cuda4/cu"
	"nimble-cube/core"
)

// Uploads data from host to GPU.
type Uploader struct {
	host   core.RChan
	dev    Chan
	bsize  int
	stream cu.Stream
}

func NewUploader(hostdata core.RChan, devdata Chan) *Uploader {
	core.Assert(hostdata.Size() == devdata.Size())
	return &Uploader{hostdata, devdata, core.BlockSize(hostdata.Size()), 0}
}

func (u *Uploader) Run() {
	core.Debug("uploader: run")
	LockCudaThread()
	defer UnlockCudaThread()
	u.stream = cu.StreamCreate()
	MemHostRegister(u.host.UnsafeData())
	bsize := u.bsize

	for {
		in := u.host.ReadNext(bsize)
		out := u.dev.WriteNext(bsize)
		core.Debug("upload", bsize)
		out.CopyHtoDAsync(in, u.stream)
		u.stream.Synchronize()
		u.host.ReadDone()
		u.dev.WriteDone()
	}
}
