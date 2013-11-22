package gui

type E struct {
	dirty bool
	el    El
}

func (e *E) set(v interface{}) {
	old := e.el.value() // carefully check if value changed, set/value may do things behind the screens
	e.el.set(v)
	if e.el.value() != old {
		e.dirty = true
	}
}

type El interface {
	update(id string) jsCall
	set(v interface{})
	value() interface{}
}
