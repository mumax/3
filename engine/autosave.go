package engine

// Bookkeeping for auto-saving quantities at given intervals.

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/draw"
	"github.com/mumax/3/util"
)

var (
	output  = make(map[Quantity]*autosave) // when to save quantities
	autonum = make(map[interface{}]int)    // auto number for out file
)

func init() {
	DeclFunc("Save", Save, "Save space-dependent quantity once, with auto filename")
	DeclFunc("SaveAs", SaveAs, "Save space-dependent with custom filename")
	DeclFunc("AutoSave", AutoSave, "Auto save space-dependent quantity every period (s).")
	DeclVar("SnapshotFormat", &SnapshotFormat, "Image format for snapshots: jpg, png or gif.")
	DeclVar("FilenameFormat", &FilenameFormat, "printf formatting string for output filenames.")
	DeclFunc("Snapshot", Snapshot, "Save image of quantity")
	DeclFunc("AutoSnapshot", AutoSnapshot, "Auto save image of quantity every period (s).")
}

// Periodically called by run loop to save everything that's needed at this time.
func DoOutput() {
	for q, a := range output {
		if a.needSave() {
			a.save(q)
			a.count++
		}
	}
	if Table.needSave() {
		Table.Save()
	}
}

// Register quant to be auto-saved every period.
// period == 0 stops autosaving.
func AutoSave(q Quantity, period float64) {
	autoSave(q, period, Save)
}

// Register quant to be auto-saved as image, every period.
func AutoSnapshot(q Quantity, period float64) {
	autoSave(q, period, Snapshot)
}

// register save(q) to be called every period
func autoSave(q Quantity, period float64, save func(Quantity)) {
	if period == 0 {
		delete(output, q)
	} else {
		output[q] = &autosave{period, Time, 0, save}
	}
}

var (
	FilenameFormat = "%s%06d" // formatting string for auto filenames.
	SnapshotFormat = "jpg"
)

// Save once, with auto file name
func Save(q Quantity) {
	fname := autoFname(q.Name(), autonum[q])
	SaveAs(q, fname)
	autonum[q]++
}

// generate auto file name based on save count and FilenameFormat. E.g.:
// 	m000001.ovf
func autoFname(name string, num int) string {
	return fmt.Sprintf(OD+FilenameFormat+".ovf", name, num)
}

// Save image once, with auto file name
func Snapshot(q Quantity) {
	fname := fmt.Sprintf(OD+FilenameFormat+"."+SnapshotFormat, q.Name(), autonum[q])
	s, r := q.Slice()
	if r {
		defer cuda.Recycle(s)
	}
	data := s.HostCopy() // must be copy (asyncio)
	autonum[q]++
	saveQue <- func() { util.FatalErr(draw.RenderFile(fname, data, "", "")) }
}

// keeps info needed to decide when a quantity needs to be periodically saved
type autosave struct {
	period float64        // How often to save
	start  float64        // Starting point
	count  int            // Number of times it has been autosaved
	save   func(Quantity) // called to do the actual save
}

// returns true when the time is right to save.
func (a *autosave) needSave() bool {
	t := Time - a.start
	return a.period != 0 && t-float64(a.count)*a.period >= a.period
}
