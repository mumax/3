package gpu

import (
	"github.com/barnex/cuda4/cu"
	"github.com/barnex/cuda4/safe"
	"nimble-cube/core"
)

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

type Downloader struct {
	devdata  safe.Float32s
	rlock    *core.RMutex
	hostdata []float32
	wlock    *core.RWMutex
	bsize    int
	stream   cu.Stream
}

func NewDownloader(devdata safe.Float32s, devlock *core.RMutex, hostdata []float32, hostlock *core.RWMutex) *Downloader {
	u := new(Downloader)
	u.devdata = devdata
	u.rlock = devlock
	u.hostdata = hostdata
	u.wlock = hostlock
	u.bsize = 16 // TODO !! Always lock max
	return u
}

func (u *Downloader) Run() {
	core.Debug("downloader: run")
	LockCudaThread()
	u.stream = cu.StreamCreate()
	MemHostRegister(u.hostdata)
	bsize := u.bsize

	for {
		for i := 0; i < len(u.hostdata); i += bsize {
			u.rlock.ReadNext(u.bsize)
			u.wlock.WriteNext(u.bsize)
			core.Debug("download", i, bsize)
			u.devdata.CopyDtoHAsync(u.hostdata, u.stream)
			u.stream.Synchronize()
			u.rlock.ReadDone()
			u.wlock.WriteDone()
		}
	}
}
