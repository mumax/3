package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"log"
	"reflect"
)

type Box struct {
	out  *Chan
	ins  []*Reader
	exec func(out *data.Slice, in ...*data.Slice)
}

func NewBox(fn interface{}, out *Chan, in ...*Reader) *Box {
	return &Box{out, in, wrapFn(fn)}
}

func wrapFn(fn interface{}) func(*data.Slice, ...*data.Slice) {
	switch f := fn.(type) {
	default:
		log.Panicf("box: illegal func type:", reflect.TypeOf(f))
	case func(out, in *data.Slice):
		return func(out *data.Slice, in ...*data.Slice) {
			f(out, in[0])
		}
	}
	panic("unreachable")
	return nil
}

func (b *Box) NumIn() int {
	return len(b.ins)
}

func (b *Box) Exec() {
	out := b.out.WriteNext()
	ins := make([]*data.Slice, b.NumIn())
	for i := range ins {
		ins[i] = b.ins[i].ReadNext()
	}
	b.exec(out, ins...)
	for _, in := range b.ins {
		in.ReadDone()
	}
	b.out.WriteDone()
}

func (b *Box) Run() {
	cuda.LockThread()
	for {
		b.Exec()
	}
}
