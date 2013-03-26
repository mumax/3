package engine

import(
	"fmt"
)

type autosave struct {
	period float64 // How often to save
	start  float64 // Starting point
	count  int     // Number of times it has been saved
	name   string
}

func (a *autosave) Autosave(period float64) {
	a.period = period
	a.start = Time
	a.count = 0
}

func (a *autosave) needSave() bool {
	if a.period == 0 {
		return false
	}
	t := Time - a.start
	return t-float64(a.count)*a.period >= a.period
}

func (a *autosave) saved() {
	a.count++
}

func (a *autosave) fname() string {
	return fmt.Sprintf("%s%s%06d.dump", OD, a.name, a.count)
}

