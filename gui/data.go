package gui

type data struct {
	val any
}

func (d *data) set(v any)  { d.val = v }
func (d *data) value() any { return d.val }
