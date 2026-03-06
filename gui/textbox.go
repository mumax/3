package gui

import "fmt"

type textbox struct {
	data
}

func (e *textbox) update(id string) []jsCall {
	return []jsCall{{F: "setTextbox", Args: []any{id, e.value()}}}
}

func (d *Page) TextBox(id string, value any, extra ...string) string {
	e := &textbox{data: data{value}}
	d.addElem(id, e)
	return fmt.Sprintf(`<input type=%v class=TextBox id=%v  onchange="notifyel('%v', 'value')" onfocus="notifyfocus('%v')" onblur="notifyblur('%v')" onkeydown="makered('%v', event)" %v />`, "text", id, id, id, id, id, cat(extra))
}
