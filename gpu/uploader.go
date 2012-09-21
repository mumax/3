package gpu

import (
	"github.com/barnex/cuda4/cu"
	//"github.com/barnex/cuda4/safe"
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

// _____________________________________

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
