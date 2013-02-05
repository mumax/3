package gpu

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/nimble"
	"github.com/barnex/cuda5/cu"
)

// Downloads data from host to GPU.
type Downloader struct {
	host   nimble.ChanN
	dev    nimble.RChanN
	stream cu.Stream
}

func NewDownloader(tag, unit string, devdata_ nimble.ChanN) *Downloader {
	devdata := devdata_.NewReader()
	hostdata := nimble.MakeChanN(devdata.NComp(), devdata.Tag(), devdata.Unit(), devdata.Mesh(), nimble.CPUMemory, 0)
	panic("register")
	//MemHostRegister(hostdata.UnsafeData().Host())
	u := &Downloader{hostdata, devdata, cu.StreamCreate()}
	nimble.Stack(u)
	return u
}

func (u *Downloader) Output() nimble.ChanN {
	return u.host
}

func (u *Downloader) Run() {
	core.Debug("run gpu.downloader")
	N := u.host.BufLen()
	panic("lock")
	//LockCudaThread()

	for {
		for c := 0; c < u.dev.NComp(); c++ {
			dev := u.dev.Comp(c)
			host := u.host.Comp(c)
			out := host.WriteNext(N)
			in := dev.ReadNext(N)
			CopyDtoH(out, in)
			//in.CopyDtoHAsync(out, u.stream)
			//u.stream.Synchronize()
			host.WriteDone()
			dev.ReadDone()
		}
	}
}
