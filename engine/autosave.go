package engine

import "fmt"

var (
	output  = make(map[Saver]*autosave) // when to save quantities
	autonum = make(map[interface{}]int) // auto number for out file
)

func init() {
	DeclFunc("Save", Save, "Save space-dependent quantity once, with auto filename")
	DeclFunc("SaveAs", SaveAs, "Save space-dependent with custom filename")
	DeclFunc("AutoSave", AutoSave, "Auto save space-dependent quantity every period (s).")
}

// Anything that can be saved with auto name
type Saver interface {
	Save()
}

// Register quant to be auto-saved every period.
// period == 0 stops autosaving.
func AutoSave(quant Saver, period float64) {
	if period == 0 {
		delete(output, quant)
	} else {
		output[quant] = &autosave{period, Time, 0}
	}
}

// Called to save everything that's needed at this time.
func DoOutput() {
	for q, a := range output {
		if a.needSave() {
			q.Save()
			a.count++
		}
	}
}

// Save once, with auto file name
func Save(q Getter) {
	fname := fmt.Sprintf("%s%06d.dump", q.Name(), autonum[q])
	SaveAs(q, fname)
	autonum[q]++
}

// keeps info needed to decide when a quantity needs to be periodically saved
type autosave struct {
	period float64 // How often to save
	start  float64 // Starting point
	count  int     // Number of times it has been autosaved
}

// returns true when the time is right to save.
func (a *autosave) needSave() bool {
	t := Time - a.start
	return t-float64(a.count)*a.period >= a.period
}
