package gpu

import (
	"code.google.com/p/nimble-cube/nimble"
	"github.com/barnex/cuda5/cu"
)

// Uploads data from host to GPU.
type Uploader struct {
	host   nimble.RChan1
	dev    nimble.Chan1
	bsize  int
	stream cu.Stream
}

func NewUploader(hostdata nimble.RChan1, devdata nimble.Chan1) *Uploader {
	nimble.Assert(hostdata.Size() == devdata.Size())
	blocklen := nimble.Prod(nimble.BlockSize(hostdata.Size()))
	return &Uploader{hostdata, devdata, blocklen, 0}
}

func (u *Uploader) Run() {
	nimble.Debug("run gpu.uploader with block size", u.bsize)
	LockCudaThread()
	defer UnlockCudaThread()
	u.stream = cu.StreamCreate()
	MemHostRegister(u.host.UnsafeData())

	for {
		in := u.host.ReadNext(u.bsize).Host()
		out := u.dev.WriteNext(u.bsize).Device()
		out.CopyHtoDAsync(in, u.stream)
		u.stream.Synchronize()
		u.dev.WriteDone()
		u.host.ReadDone()
	}
}

func RunUploader(tag string, input nimble.Chan) nimble.ChanN {
	in := input.ChanN()

	output := nimble.MakeChanN(in.NComp(), tag, in.Unit(), in.Mesh(), nimble.GPUMemory)

	for i := range in {
		nimble.Stack(NewUploader(in[i].NewReader(), output[i]))
	}
	return output
}
