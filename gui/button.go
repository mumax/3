package gui

import "fmt"

type button struct {
	data
}

func (e *button) update(id string) []jsCall {
	return []jsCall{{F: "setAttr", Args: []interface{}{id, "innerHTML", e.value()}}}
}

func (d *Page) Button(id string, value interface{}, extra ...string) string {
	e := &button{data: data{value}}
	d.addElem(id, e)
	return fmt.Sprintf(`<button id=%v class=Button onclick="notifyel('%v', 'innerHTML')"></button>`, id, id)
}
