package engine

import "fmt"

var (
	output  = make(map[Getter]*autosave) // when to save quantities
	autonum = make(map[interface{}]int)  // auto number for out file
)

func init() {
	DeclFunc("Save", Save, "Save space-dependent quantity once, with auto filename")
	DeclFunc("SaveAs", SaveAs, "Save space-dependent with custom filename")
	DeclFunc("AutoSave", AutoSave, "Auto save space-dependent quantity every period (s).")
}

// Register quant to be auto-saved every period.
// period == 0 stops autosaving.
func AutoSave(quant Getter, period float64) {
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
			Save(q)
			a.count++
		}
	}
	if Table.needSave() {
		Table.Save()
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
	return a.period != 0 && t-float64(a.count)*a.period >= a.period
}
