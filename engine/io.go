package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"path"
	"strings"
)

// TODO: only use getter, check if slice is on GPU?
type GPU_Getter interface {
	GetGPU() (q *data.Slice, recycle bool) // get quantity data (GPU), indicate need to recycle
}

type Getter interface {
	Get() (q *data.Slice, recycle bool) // get quantity data (GPU or CPU), indicate need to recycle
}

type Saver interface {
	Getter
	autoFname() string // file name for autosave
	needSave() bool
	saved()
}

type Autosaver interface {
	Autosave(period float64)
}

// Download a quantity to host,
// or just return its data when already on host.
func Download(q Getter) *data.Slice {
	buf, recycle := q.Get()
	if buf.CPUAccess() {
		util.Assert(recycle == false)
		return buf
	}
	h := hostBuf(buf.NComp(), buf.Mesh())
	data.Copy(h, buf)
	if recycle {
		cuda.RecycleBuffer(buf)
	}
	return h
}

func hostBuf(nComp int, m *data.Mesh) *data.Slice {
	return data.NewSlice(nComp, m) // TODO use pool of page-locked buffers
}

func SaveAs(q Getter, fname string) {
	if !path.IsAbs(fname) && !strings.HasPrefix(fname, OD) {
		fname = path.Clean(OD + "/" + fname)
	}
	if path.Ext(fname) == "" {
		fname += ".dump"
	}
	buffer, recylce := q.Get()
	if recylce {
		defer cuda.RecycleBuffer(buffer)
	}
	AsyncSave(fname, assureCPU(buffer), Time)
}

func assureGPU(s *data.Slice) *data.Slice {
	if s.GPUAccess() {
		return s
	} else {
		return cuda.GPUCopy(s)
	}
}

func assureCPU(s *data.Slice) *data.Slice {
	if s.CPUAccess() {
		return s
	} else {
		return s.HostCopy()
	}
}
