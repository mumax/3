package gui

import "fmt"

type span struct {
	data
}

func (e *span) update(id string) []jsCall {
	return []jsCall{{F: "setAttr", Args: []interface{}{id, "innerHTML", e.value()}}}
}

// {{.Span id value}} adds a piece of text ("label") to the document.
func (d *Page) Span(id string, value interface{}, extra ...string) string {
	e := &span{data: data{value}}
	d.addElem(id, e)
	return fmt.Sprintf(`<span id=%v %v> </span>`, id, cat(extra))
}
