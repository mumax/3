package conv

import (
	"github.com/barnex/cuda4/cu"
	"nimble-cube/core"
	"nimble-cube/gpu"
)

// Uploader for vector data specifically tuned for convolution.
// Prioritizes upload of all X components, then Y, then Z.
type Uploader struct {
	host   core.RChan3
	dev    [3]gpu.Chan
	bsize  int
	stream cu.Stream
}

//func NewUploader(hostdata core.RChan3, devdata [3]Chan) *Uploader {
//	core.Assert(hostdata.Size() == devdata.Size())
//	blocklen := core.Prod(core.BlockSize(hostdata.Size()))
//	return &Uploader{hostdata, devdata, blocklen, 0}
//}

func (u *Uploader) Run() {
	//	core.Debug("run gpu.uploader with block size", u.bsize)
	//	LockCudaThread()
	//	defer UnlockCudaThread()
	//	u.stream = cu.StreamCreate()
	//	MemHostRegister(u.host.UnsafeData())
	//
	//	for {
	//		in := u.host.ReadNext(u.bsize)
	//		out := u.dev.WriteNext(u.bsize)
	//		out.CopyHtoDAsync(in, u.stream)
	//		u.stream.Synchronize()
	//		u.host.ReadDone()
	//		u.dev.WriteDone()
	//	}
}
