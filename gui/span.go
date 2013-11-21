package gui

import (
	"fmt"
)

type span struct {
}

// {{.Span id value}} adds a piece of text ("label") to the document.
func (d *Page) Span(id string, value string, extra ...string) string {
	e := &span{}
	d.addElem(id, e)

	//	el.setValue(value)
	//	el.update = func(id string) jsCall {
	//		return jsCall{F: "setAttr", Args: []interface{}{id, "innerHTML", el.value()}}
	//	}

	return fmt.Sprintf(`<span id=%v %v>%v</span>`, id, cat(extra), value)
}
