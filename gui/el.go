package gui

type E struct {
	dirty bool
	_attr map[string]interface{}
	_elem El
}

func newE(elem El) *E {
	return &E{_elem: elem, dirty: true}
}

func (e *E) set(v interface{}) {
	old := e.value() // carefully check if value changed, set/value may do things behind the screens
	e._elem.set(v)
	if e.value() != old {
		e.dirty = true
	}
}

func (e *E) attr(key string, v interface{}) {
	if e._attr == nil {
		e._attr = make(map[string]interface{})
	}
	old := e._attr[key]
	if v != old {
		e.dirty = true
	}
	e._attr[key] = v
}

func (e *E) update(id string) []jsCall {
	upd := e._elem.update(id)
	for k, v := range e._attr {
		upd = append(upd, jsCall{F: "setAttr", Args: []interface{}{id, k, v}})
	}
	return upd
}

func (e *E) value() interface{} {
	return e._elem.value()
}

type El interface {
	update(id string) []jsCall
	set(v interface{})
	value() interface{}
}
