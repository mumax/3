package gui

import "sync"

// wraps a GUI element (button, textbox, ...),
// stores the dirty flag, extra attributes, lock event handler, ...
type E struct {
	_m      sync.Mutex
	_dirty  bool                   // dirty means the value/attributes need updating in browser
	_attr   map[string]interface{} // extra html attributes (e.g. style, onclick, ...)
	_elem   El                     // the wrapped gui element
	onevent func()                 // called upon value change by user (not by Go code)
}

func newE(elem El) *E {
	return &E{_elem: elem, _dirty: true}
}

// atomically pass a new value to the underlying element and mark it dirty.
func (e *E) set(v interface{}) {
	e._m.Lock()
	defer e._m.Unlock()

	old := e._elem.value() // carefully check if value changed, set/value may do things behind the screens
	e._elem.set(v)
	if e._elem.value() != old {
		e._dirty = true
	}
}

// atomically set an html attribute for the underlying element and mark it dirty
func (e *E) attr(key string, v interface{}) {
	e._m.Lock()
	defer e._m.Unlock()

	if e._attr == nil {
		e._attr = make(map[string]interface{})
	}
	old := e._attr[key]
	if v != old {
		e._dirty = true
	}
	e._attr[key] = v
}

// atomically produce a list of javascript calls needed to update the element in the browser,
// and clear dirty flag
func (e *E) update(id string) []jsCall {
	e._m.Lock()
	defer e._m.Unlock()
	if !e._dirty {
		return []jsCall{}
	}
	upd := e._elem.update(id)
	for k, v := range e._attr {
		upd = append(upd, jsCall{F: "setAttr", Args: []interface{}{id, k, v}})
	}
	e._dirty = false
	return upd
}

// atomically returns the underlying element's value
// depending its implementation (e.g. textBox's text, checkBox's checked value, etc.)
func (e *E) value() interface{} {
	e._m.Lock()
	defer e._m.Unlock()

	return e._elem.value()
}

// atomically set the dirty flag w/o changing value.
// called, e.g., when a second brower window opens
func (e *E) setDirty() {
	e._m.Lock()
	defer e._m.Unlock()
	e._dirty = true
}

// Atomically set a new onevent function, which is called each time
// the user changes the underlying elements value.
func (e *E) OnEvent(f func()) {
	e._m.Lock()
	defer e._m.Unlock()

	e.onevent = f
}

// Underlying html element like Span, TextBox, etc.
type El interface {
	update(id string) []jsCall
	set(v interface{})
	value() interface{}
}
