package gui

import (
	"fmt"
)

type textbox struct {
	data
}

func (e *textbox) update(id string) []jsCall {
	return []jsCall{{F: "setAttr", Args: []interface{}{id, "value", e.value()}}}
}

func (d *Page) TextBox(id string, value interface{}, extra ...string) string {
	e := &textbox{data: data{value}}
	d.addElem(id, e)
	return fmt.Sprintf(`<input type=%v class=TextBox id=%v onchange="notifytextbox('%v')" onfocus="notifyfocus('%v')" onblur="notifyblur('%v')" %v />`, "text", id, id, id, id, cat(extra))
}
