package gui

import "fmt"

type number struct {
	data
}

func (e *number) update(id string) []jsCall {
	return []jsCall{{F: "setAttr", Args: []interface{}{id, "value", e.value()}}}
}

func (d *Page) Number(id string, min, max, value int, extra ...string) string {
	e := &number{data: data{value}}
	d.addElem(id, e)
	return fmt.Sprintf(`<input type=%v class=TextBox id=%v onchange="notifyel('%v', 'value')" onfocus="notifyfocus('%v')" onblur="notifyblur('%v')" min=%v max=%v %v />`, "number", id, id, id, id, min, max, cat(extra))
}
