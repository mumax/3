package engine

type autosaver struct {
	period float64 // How often to save
	start  float64 // Starting point
	count  int     // Number of times it has been saved
}

func (a *autosaver) notify() {
	t := Time - a.start
	if t-float64(a.count)*a.period >= a.period {
		//SaveAs(e.Quant(a.quant), a.format, a.options, e.AutoFilename(a.quant, a.format))
		a.count++
	}
}
