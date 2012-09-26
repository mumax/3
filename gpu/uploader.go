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
	blocklen := core.Prod(core.BlockSize(hostdata.Size()))
	return &Uploader{hostdata, devdata, blocklen, 0}
}

func (u *Uploader) Run() {
	core.Debug("run gpu.uploader with block size", u.bsize)
	LockCudaThread()
	defer UnlockCudaThread()
	u.stream = cu.StreamCreate()
	MemHostRegister(u.host.UnsafeData())

	for {
		in := u.host.ReadNext(u.bsize)
		out := u.dev.WriteNext(u.bsize)
		out.CopyHtoDAsync(in, u.stream)
		u.stream.Synchronize()
		u.dev.WriteDone()
		u.host.ReadDone()
	}
}
