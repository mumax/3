package gui

import "fmt"

func (d *Doc) Range(id string, min, max, value int) string {
	e := newElem(id, "checked", value)
	d.add(id, e)
	return fmt.Sprintf(`<input type=range id=%v min=%v max=%v value=%v onchange="notifyrange('%v')"/>`, id, min, max, value, id)
}
