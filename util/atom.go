package util

import "sync/atomic"

// Atomic int
type Atom int32

func (a *Atom) Add(v int32) {
	atomic.AddInt32((*int32)(a), v)
}

func (a *Atom) Load() int32 {
	return atomic.LoadInt32((*int32)(a))
}
