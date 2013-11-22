package gui

type E struct {
	dirty bool
	el    El
}

func (e *E) set(v interface{}) {
	e.el.set(v)
	e.dirty = true
}

type El interface {
	update(id string) jsCall
	set(v interface{})
}
