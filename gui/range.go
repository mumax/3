package gui

import "fmt"

type slider struct {
	data
}

func (e *slider) update(id string) []jsCall {
	return []jsCall{{F: "setAttr", Args: []interface{}{id, "value", e.value()}}}
}

func (d *Page) Range(id string, min, max, value int, extra ...string) string {
	e := &slider{data: data{value}}
	d.addElem(id, e)
	return fmt.Sprintf(`<input type=range id=%v min=%v max=%v oninput="notifyel('%v', 'value')" onchange="notifyel('%v', 'value')"/>`, id, min, max, id)
}
