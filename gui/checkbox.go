package gui

import "fmt"

type checkbox struct {
	data
}

func (e *checkbox) update(id string) []jsCall {
	return []jsCall{{F: "setAttr", Args: []interface{}{id, "checked", e.value()}}}
}

func (d *Page) Checkbox(id, text string, value bool, extra ...string) string {
	e := &checkbox{data: data{value}}
	d.addElem(id, e)
	return fmt.Sprintf(`<input type=checkbox id=%v class=CheckBox onchange="notifyel('%v', 'checked')">%v</input>`, id, id, text)
}
