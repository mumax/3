package engine

// Bookkeeping for auto-saving quantities at given intervals.

import "fmt"

var (
	output  = make(map[Quantity]*autosave) // when to save quantities
	autonum = make(map[interface{}]int)    // auto number for out file
)

func init() {
	DeclFunc("AutoSave", AutoSave, "Auto save space-dependent quantity every period (s).")
	DeclFunc("AutoSnapshot", AutoSnapshot, "Auto save image of quantity every period (s).")
	DeclFunc("AutoSaveSteps", AutoSaveStep, " Auto save space-dependent quantity every Nsteps.")
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

func AutoSaveStep(q Quantity, nbr_steps int) {
	autoSave_steps(q, nbr_steps, Save)
}

// Register quant to be auto-saved as image, every period.
func AutoSnapshot(q Quantity, period float64) {
	autoSave(q, period, Snapshot)
}

// register save(q) to be called every period
func autoSave(q Quantity, period float64, save func(Quantity)) {
	nbr_steps := 0
	if period == 0 {
		delete(output, q)
	} else {
		output[q] = &autosave{period, nbr_steps, Time, -1, save} // init count to -1 allows save at t=0
	}
}

func autoSave_steps(q Quantity, nbr_steps int, save func(Quantity)) {
	period := 0.0
	if nbr_steps == 0 {
		delete(output, q)
	} else {
		output[q] = &autosave{period, nbr_steps, Time, -1, save} // init count to -1 allows save at t=0
	}
}

// generate auto file name based on save count and FilenameFormat. E.g.:
// 	m000001.ovf
func autoFname(name string, num int) string {
	return fmt.Sprintf(OD()+FilenameFormat+".ovf", name, num)
}

// keeps info needed to decide when a quantity needs to be periodically saved
type autosave struct {
	period    float64        // How often to save
	nbr_steps int            // save after given nbr of steps
	start     float64        // Starting point
	count     int            // Number of times it has been autosaved
	save      func(Quantity) // called to do the actual save
}

// returns true when the time is right to save.
func (a *autosave) needSave() bool {
	if a.nbr_steps == 0 {
		t := Time - a.start
		return a.period != 0 && t-float64(a.count)*a.period >= a.period
	} else {
		nbr := NSteps - a.nbr_steps
		return a.nbr_steps != 0 && nbr-a.count*a.nbr_steps >= a.nbr_steps
	}
}
