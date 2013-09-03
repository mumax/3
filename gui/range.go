package gui

import (
	"fmt"
)

func (t *Templ) Range(id string, min, max, value int) string {
	d := (*Doc)(t)
	el := d.addElem(id)
	el.data = &intData{value}
	el.setValue(value)
	el.update = func(id string) jsCall {
		return jsCall{F: "setAttr", Args: []interface{}{id, "value", el.value().(int)}}
	}
	return fmt.Sprintf(`<input type=range id=%v min=%v max=%v value=%v onchange="notifyrange('%v')"/>`, id, min, max, value, id)
}
