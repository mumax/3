package gui

import "fmt"

type textarea struct {
	data
}

func (e *textarea) update(id string) []jsCall {
	return []jsCall{{F: "setAttr", Args: []interface{}{id, "innerHTML", e.value()}}}
}

func (d *Page) TextArea(id string, rows, cols int, value interface{}, extra ...string) string {
	e := &textarea{data: data{value}}
	d.addElem(id, e)
	return fmt.Sprintf(`<textarea id=%v rows=%v cols=%v class=TextBox %v></textarea>`, id, rows, cols, cat(extra))
}
