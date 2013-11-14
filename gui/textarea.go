package gui

import (
	"fmt"
)

func (t *Templ) TextArea(id string, rows, cols int, value string) string {
	d := (*Doc)(t)
	el := d.addElem(id)
	el.setValue(value)
	el.update = func(id string) jsCall {
		return jsCall{F: "setAttr", Args: []interface{}{id, "value", el.value()}}
	}
	return fmt.Sprintf(`<textarea id=%v rows=%v cols=%v>%v</textarea>`, id, rows, cols, value)
}
