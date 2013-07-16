package gui

import (
	"fmt"
)

type Button struct {
	elem
}

func (d *Doc) Button(id, value string) string {
	e := &Button{makeElem(id, value)}
	d.add(e)
	return e.Render()
}

func (e *Button) Render() string {
	return fmt.Sprintf(`<button id=%v onclick="call('%v')">%v</button>`, e.id, e.id, "")
}
