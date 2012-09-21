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
	return &Uploader{hostdata, devdata, 16, 0} // TODO: block size
}

func (u *Uploader) Run() {
	core.Debug("uploader: run")
	LockCudaThread()
	defer UnlockCudaThread()
	u.stream = cu.StreamCreate()
	MemHostRegister(u.host.List)
	bsize := u.bsize

	for {
		for i := 0; i < len(u.host.List); i += bsize {
			u.host.ReadNext(bsize)
			u.dev.WriteNext(bsize)
			core.Debug("upload", i, bsize)
			u.dev.CopyHtoDAsync(u.host.List, u.stream)
			u.stream.Synchronize()
			u.host.ReadDone()
			u.dev.WriteDone()
		}
	}
}
