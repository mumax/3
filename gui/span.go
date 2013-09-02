package gui

import (
	"fmt"
)

// {{.Span id value}} adds a piece of text ("label") to the document.
func (t *Templ) Span(id string, value ...string) string {
	d := (*Doc)(t)
	val := cat(value)
	el := d.addElem(id)
	el.setValue(val)
	el.update = func(id string) jsCall {
		return jsCall{F: "setAttr", Args: []interface{}{id, "innerHTML", el.value()}}
	}
	return fmt.Sprintf(`<span id=%v>%v</span>`, id, val)
}
