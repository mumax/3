package gui

import (
	"fmt"
)

type span struct {
	id    string
	value interface{}
}

func (e *span) set(v interface{}) {
	e.value = v
}

func (e *span) update(id string) jsCall {
	return jsCall{F: "setAttr", Args: []interface{}{id, "innerHTML", e.value}}
}

// {{.Span id value}} adds a piece of text ("label") to the document.
func (d *Page) Span(id string, value interface{}, extra ...string) string {
	e := &span{id: id, value: value}
	d.addElem(id, e)

	//	el.setValue(value)
	//	el.update = func(id string) jsCall {
	//		return jsCall{F: "setAttr", Args: []interface{}{id, "innerHTML", el.value()}}
	//	}

	return fmt.Sprintf(`<span id=%v %v> </span>`, id, cat(extra))
}
