package gui

import "fmt"

type element struct {
	data
}

func (e *element) update(id string) []jsCall {
	return []jsCall{{F: "setAttr", Args: []interface{}{id, "innerHTML", e.value()}}}
}

func (d *Page) Element(id, typ, attr string, value interface{}, extra ...string) string {
	e := &element{data: data{value}}
	d.addElem(id, e)
	return fmt.Sprintf(`<span id=%v %v> </span>`, id, cat(extra))
}
