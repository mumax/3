package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"fmt"
	"os"
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
	if s, ok := q.(GPU_Getter); ok {
		buffer, recylce := s.GetGPU()
		if recylce {
			defer cuda.RecycleBuffer(buffer)
		}
		goSaveCopy(fname, buffer, Time)
	} else {
		h, recycle := q.Get()
		// if this assertion fails, it means q's Get() returns a GPU slice,
		// yet, it does not implement GPU_Getter. So, add GetGPU() to that type!
		util.Assert(recycle == false)
		data.MustWriteFile(fname, h, Time) // not async, but only for stuff already on CPU. could be improved
	}
}

func assureGPU(s *data.Slice) *data.Slice {
	if s.GPUAccess() {
		return s
	} else {
		return cuda.GPUCopy(s)
	}
}

// Append msg to file. Used to write aggregated output of many simulations in one file.
func Fprintln(filename string, msg ...interface{}) {
	f, err := os.OpenFile(filename, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0666)
	util.FatalErr(err)
	defer f.Close()
	_, err = fmt.Fprintln(f, msg...)
	util.FatalErr(err)
}
