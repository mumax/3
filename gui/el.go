package gui

import "sync"

type E struct {
	dirty   bool
	_attr   map[string]interface{}
	_elem   El
	_m      sync.Mutex
	onevent func()
}

func newE(elem El) *E {
	return &E{_elem: elem, dirty: true}
}

func (e *E) set(v interface{}) {
	e._m.Lock()
	defer e._m.Unlock()

	old := e._elem.value() // carefully check if value changed, set/value may do things behind the screens
	e._elem.set(v)
	if e._elem.value() != old {
		e.dirty = true
	}
}

func (e *E) attr(key string, v interface{}) {
	e._m.Lock()
	defer e._m.Unlock()

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
	e._m.Lock()
	defer e._m.Unlock()

	upd := e._elem.update(id)
	for k, v := range e._attr {
		upd = append(upd, jsCall{F: "setAttr", Args: []interface{}{id, k, v}})
	}
	return upd
}

func (e *E) value() interface{} {
	e._m.Lock()
	defer e._m.Unlock()

	return e._elem.value()
}

func (e *E) OnEvent(f func()) {
	e._m.Lock()
	defer e._m.Unlock()

	e.onevent = f
}

type El interface {
	update(id string) []jsCall
	set(v interface{})
	value() interface{}
}
