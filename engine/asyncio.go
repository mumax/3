package engine

// Asynchronous I/O queue flushes data to disk while simulation keeps running.

import (
	"bufio"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/oommf"
	"github.com/mumax/3/util"
	"io"
	"os"
	"path"
	"strings"
)

func init() {
	DeclVar("OutputFormat", &outputformat, "Format for data files: OVF1_TEXT, OVF1_BINARY, OVF2_TEXT or OVF2_BINARY")
	DeclROnly("OVF1_BINARY", OVF1_BINARY, "OutputFormat = OVF1_BINARY sets binary OVF1 output")
	DeclROnly("OVF2_BINARY", OVF2_BINARY, "OutputFormat = OVF2_BINARY sets binary OVF2 output")
	DeclROnly("OVF1_TEXT", OVF1_TEXT, "OutputFormat = OVF1_TEXT sets text OVF1 output")
	DeclROnly("OVF2_TEXT", OVF2_TEXT, "OutputFormat = OVF2_TEXT sets text OVF2 output")
}

// Save under given file name (transparent async I/O).
func SaveAs(q Quantity, fname string) {
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
	info := data.Meta{Time: Time, Name: q.Name(), Unit: q.Unit(), CellSize: q.Mesh().CellSize()}
	initQue()
	data := buffer.HostCopy() // must be copy (async io)
	saveQue <- func() { saveFile(fname, data, info) }
}

var (
	saveQue      chan func()       // passes save requests from runDownloader to runSaver
	done         = make(chan bool) // marks output server is completely done after closing dlQue
	nOutBuf      int               // number of output buffers actually in use (<= maxOutputQueLen)
	outputformat = OVF2_BINARY     // user-settable output format
)

const maxOutputQueLen = 16 // number of outputs that can be queued for asynchronous I/O.

type OutputFormat int

const (
	OVF1_TEXT OutputFormat = iota + 1
	OVF1_BINARY
	OVF2_TEXT
	OVF2_BINARY
)

func initQue() {
	if saveQue == nil {
		saveQue = make(chan func())
		go runSaver()
	}
}

// continuously takes save tasks and flushes them to disk.
// the save queue can accommodate many outputs (stored on host).
// the rather big queue allows big output bursts to be concurrent with GPU.
func runSaver() {
	for f := range saveQue {
		f()
	}
	done <- true
}

// synchronous save
func saveFile(fname string, output *data.Slice, info data.Meta) {
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	util.FatalErr(err)
	out := bufio.NewWriter(f)
	save(out, output, info)
	out.Flush()
	f.Close()
}

// synchronous save
func save(out io.Writer, s *data.Slice, info data.Meta) {
	switch outputformat {
	case OVF1_TEXT:
		oommf.WriteOVF1(out, s, info, "text")
	case OVF1_BINARY:
		oommf.WriteOVF1(out, s, info, "binary 4")
	case OVF2_TEXT:
		oommf.WriteOVF2(out, s, info, "text")
	case OVF2_BINARY:
		oommf.WriteOVF2(out, s, info, "binary 4")
	default:
		panic("invalid output format")
	}
}

// finalizer function called upon program exit.
// waits until all asynchronous output has been saved.
func drainOutput() {
	if saveQue != nil {
		close(saveQue)
		<-done
	}
}
