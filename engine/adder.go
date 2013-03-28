package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
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

func (a *Adder) AddTo(Dst *Buffered) {
	if a.needSave() {
		Buf := AddBuf()
		buf := Buf.Write()
		cuda.Zero(buf)
		a.addFn(buf)
		dst := Dst.Write()
		cuda.Madd2(dst, dst, buf, 1, 1)
		Dst.WriteDone()
		GoSave(a.fname(), dst, Time, func() { Buf.WriteDone() }) // Buf only unlocked here to avoid overwite by next AddTo
		a.saved()
	} else {
		dst := Dst.Write()
		a.addFn(dst)
		Dst.WriteDone()
	}
}

var addBuf *Buffered

func AddBuf() *Buffered {
	if addBuf == nil {
		util.DashExit()
		log.Println("allocating GPU buffer for output")
		addBuf = NewBuffered(3, "buffer")
	}
	return addBuf
}
