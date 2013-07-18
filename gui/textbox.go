package gui

import (
	"fmt"
	"strconv"
)

// {{.Textbox id value}} adds a textbox to the document.
// value is the initial text in the box.
// optional width?
func (d *Doc) TextBox(id string, value ...string) string {
	val := cat(value)
	e := newElem(id, "value", val)
	d.add(e)
	return fmt.Sprintf(`<input type=textbox class=TextBox id=%v value="%v" onchange="notifytextbox('%v')" onfocus="notifyfocus('%v')" onblur="notifyblur('%v')"/>`, id, val, id, id, id)
}

func (d *Doc) NumBox(id string, value float64) string {
	e := newElem(id, "value", value)
	e.setValue = setNumBox // setvalue override
	d.add(e)
	return fmt.Sprintf(`<input type=textbox class=TextBox id=%v value="%v" size=10 onchange="notifytextbox('%v')" onfocus="notifyfocus('%v')" onblur="notifyblur('%v')"/>`, id, value, id, id, id)
}

func (d *Doc) IntBox(id string, value int) string {
	e := newElem(id, "value", value)
	e.setValue = setIntBox // setvalue override
	d.add(e)
	return fmt.Sprintf(`<input type=textbox class=TextBox id=%v value="%v" size=10 onchange="notifytextbox('%v')" onfocus="notifyfocus('%v')" onblur="notifyblur('%v')"/>`, id, value, id, id, id)
}

func setNumBox(e *Elem, v interface{}) {
	switch concrete := v.(type) {
	default:
		setNumBox(e, fmt.Sprint(v))
	case float64:
		e.value = concrete
	case string:
		n, err := strconv.ParseFloat(concrete, 64)
		if err == nil {
			e.value = n
		} // else: keep old value // TODO: log err to gui
	}
}

func setIntBox(e *Elem, v interface{}) {
	switch concrete := v.(type) {
	default:
		setIntBox(e, fmt.Sprint(v))
	case int:
		e.value = concrete
	case string:
		n, err := strconv.Atoi(concrete)
		if err == nil {
			e.value = n
		} // else: keep old value
	}
}
