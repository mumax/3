package engine

import (
	"bufio"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"os"
	"path"
	"strings"
)

// Any space-dependent quantity
type Slicer interface {
	Slice() (q *data.Slice, recycle bool) // get quantity data (GPU or CPU), indicate need to recycle
	NComp() int
	Name() string
	Unit() string
	Mesh() *data.Mesh
}

// Save under given file name (transparant async I/O).
func SaveAs(q Slicer, fname string) {
	if !path.IsAbs(fname) && !strings.HasPrefix(fname, OD) {
		fname = path.Clean(OD + "/" + fname)
	}
	if path.Ext(fname) == "" {
		fname += ".ovf"
	}
	buffer, recylce := q.Slice()
	if recylce {
		defer cuda.Recycle(buffer)
	}
	info := data.Meta{Time: Time, Name: q.Name(), Unit: q.Unit()}
	initQue()
	saveQue <- saveTask{fname, assureCPU(buffer), info}
}

// Copy to CPU, if needed
func assureCPU(s *data.Slice) *data.Slice {
	if s.CPUAccess() {
		return s
	} else {
		return s.HostCopy()
	}
}

var (
	saveQue chan saveTask     // passes save requests from runDownloader to runSaver
	done    = make(chan bool) // marks output server is completely done after closing dlQue
	nOutBuf int               // number of output buffers actually in use (<= maxOutputQueLen)
)

const maxOutputQueLen = 16 // number of outputs that can be queued for asynchronous I/O.

func initQue() {
	if saveQue == nil {
		saveQue = make(chan saveTask)
		go runSaver()
	}
}

// output task
type saveTask struct {
	fname  string
	output *data.Slice
	info   data.Meta
}

// continuously takes save tasks and flushes them to disk.
// the save queue can accommodate many outputs (stored on host).
// the rather big queue allows big output bursts to be concurrent with GPU.
func runSaver() {
	for t := range saveQue {
		f, err := os.OpenFile(t.fname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
		util.FatalErr(err)
		out := bufio.NewWriter(f)
		data.DumpOvf2(out, t.output, "binary", t.info)
		out.Flush()
		f.Close()
	}
	done <- true
}

// finalizer function called upon program exit.
// waits until all asynchronous output has been saved.
func drainOutput() {
	if saveQue != nil {
		close(saveQue)
		<-done
	}
}
