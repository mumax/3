package gui

import (
	"fmt"
)

type span struct {
	val interface{}
}

func (e *span) set(v interface{}) {
	e.val = v
}

func (e *span) value() interface{} {
	return e.val
}

func (e *span) update(id string) jsCall {
	return jsCall{F: "setAttr", Args: []interface{}{id, "innerHTML", e.value()}}
}

// {{.Span id value}} adds a piece of text ("label") to the document.
func (d *Page) Span(id string, value interface{}, extra ...string) string {
	e := &span{val: value}
	d.addElem(id, e)
	return fmt.Sprintf(`<span id=%v %v> </span>`, id, cat(extra))
}
