package gpu

import (
	"github.com/barnex/cuda4/cu"
	"nimble-cube/core"
)

// Downloads data from GPU to host.
type Downloader struct {
	dev    RChan
	host   core.Chan
	bsize  int
	stream cu.Stream
}

func NewDownloader(devdata RChan, hostdata core.Chan) *Downloader {
	return &Downloader{devdata, hostdata, 16, 0} // TODO: block size
}

func (u *Downloader) Run() {
	core.Debug("downloader: run")
	LockCudaThread()
	u.stream = cu.StreamCreate()
	MemHostRegister(u.host.List)
	bsize := u.bsize

	for {
		for i := 0; i < len(u.host.List); i += bsize {
			u.dev.ReadNext(u.bsize)
			u.host.WriteNext(u.bsize)
			core.Debug("download", i, bsize)
			u.dev.CopyDtoHAsync(u.host.List, u.stream)
			u.stream.Synchronize()
			u.dev.ReadDone()
			u.host.WriteDone()
		}
	}
}
