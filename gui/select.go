package gui

import "fmt"

// {{.Select id}} adds a drop-down list to the document.
// options still have to be added with ...
func (d *Doc) Select(id string) string {
	e := newElem(id, "select", 0) // select is dummy attr, signals AddOption is OK
	d.add(id, e)
	return fmt.Sprintf(`<select id=%v onchange="notifyselect('%v')"> </select>`,
		id, id) // set button value does not work
}

func (d *Doc) AddOption(id string, option ...string) {
	e := d.Elem(id)
	if e.domAttr != "select" {
		panic("AddOption can only be used on Select elements")
	}
	for _, o := range option {
		d.Call("addOption", id, o)
	}
}
