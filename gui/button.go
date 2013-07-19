package gui

import "fmt"

// {{.Button id value}} adds a button to the document.
// value is text on the button.
func (d *Doc) Button(id string, value ...string) string {
	val := cat(value)
	if val == "" {
		val = id
	}
	e := newElem(id, "value", val)
	d.add(id, e)
	return fmt.Sprintf(`<button id=%v class=Button onclick="notify('%v', 'click')">%v</button>`,
		id, id, htmlEsc(val)) // set button value does not work
}
