package nc

import "sync/atomic"

type Refcount int32

// Increment the reference count by count.
func (r *Refcount) Add(count int32) {
	atomic.AddInt32((*int32)(r), count)
}

// Return the reference count.
func (r *Refcount) Load() (count int32) {
	return atomic.LoadInt32((*int32)(r))
}
