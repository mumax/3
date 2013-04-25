package engine

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
)

type Magnetization struct {
	*buffered //  TODO: buffered <-> settable?
}

func (b *buffered) SetFile(fname string) {
	util.FatalErr(b.setFile(fname))
}

func (b *buffered) setFile(fname string) error {
	m, _, err := data.ReadFile(fname)
	if err != nil {
		return err
	}
	b.Set(m)
	return nil
}
