package gui

import (
	"fmt"
)

type data struct {
	val interface{}
}

func (d *data) set(v interface{})  { d.val = v }
func (d *data) value() interface{} { return d.val }

type stringData struct {
	data
}

func (d *stringData) set(v interface{}) {
	d.val = fmt.Sprint(v)
}
