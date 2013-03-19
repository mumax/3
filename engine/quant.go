package engine

type Quant struct {
	addTo func(dst *data.Slice) // adds quantity to dst
	autosave
}

func (q *Quant) AddTo(dst *data.Slice) {
	// if need output:
	// add to zeroed buffer, output buffer (async), add buffer to dst
	// pipe buffers to/from output goroutine
	q.addTo(dst)
}

type autosave struct {
	period float64 // How often to save
	start  float64 // Starting point
	count  int     // Number of times it has been saved
}

func (a *autosave) needSave() {
	if a.period == 0 {
		return false
	}
	t := Time - a.start
	if t-float64(a.count)*a.period >= a.period {
		a.count++
		return true
	}
	return false
}
