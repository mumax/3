package gui

import "fmt"

type clibox struct {
	data
}

func (e *clibox) update(id string) []jsCall {
	return []jsCall{{F: "setAttr", Args: []interface{}{id, "value", e.value()}}}
}

func (d *Page) CliBox(id string, value interface{}, extra ...string) string {
	e := &clibox{data: data{value}}
	d.addElem(id, e)
	return fmt.Sprintf(`<input type=%v class=TextBox id=%v  onfocus="notifyfocus('%v')" onblur="notifyblur('%v')" %v />`, "text", id, id, id, cat(extra))
}
