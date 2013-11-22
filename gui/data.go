package gui

type data struct {
	val interface{}
}

func (d *data) set(v interface{})  { d.val = v }
func (d *data) value() interface{} { return d.val }
