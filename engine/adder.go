package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"log"
)

type AddFn func(dst *data.Slice)

type Adder struct {
	addFn AddFn // calculates quantity and add result to dst
	autosave
}

func NewAdder(name string, f AddFn) *Adder {
	a := new(Adder)
	a.addFn = f
	a.name = name
	return a
}

func (a *Adder) AddTo(b *Buffered) {
	if a.needSave() {
		buf := AddBuf()
		dst := buf.Write()
		cuda.Zero(dst)
		a.addFn(dst)
		buf.WriteDone()

		//out := buf.Read()
		go func() {
			log.Println("save", b.fname())
			buf.ReadDone()
		}()
	} else {
		dst := b.Write()
		a.addFn(dst)
		b.WriteDone()
	}
}

var addBuf *Buffered

func AddBuf() *Buffered {
	if addBuf == nil {
		log.Println("allocating buffer for output")
		addBuf = NewBuffered(3, "buffer")
	}
	return addBuf
}
