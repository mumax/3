package gui

import (
	"fmt"
)

// {{.Span id value}} adds a piece of text ("label") to the document.
func (d *Doc) Span(id string, value ...string) string {
	val := ""
	for _, v := range value {
		val += v + " "
	}
	e := newElem(id, "innerHTML", val)
	d.add(e)
	return fmt.Sprintf(`<span id=%v>%v</span>`, e.Id(), val)
}
