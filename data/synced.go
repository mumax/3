package data

import (
	"sync"
)

// Slice + RWMutex combination.
type Synced struct {
	buffer Slice
	lock   sync.RWMutex
}

// NewSynced wraps the slice with an RWMutex. After construction, the Slice should only be used through the Synced anymore.
func NewSynced(slice *Slice) *Synced {
	s := new(Synced)
	s.buffer = *slice
	return s
}

// Locks for reading and returns the (now locked) data Slice.
func (b *Synced) Read() *Slice {
	b.lock.RLock()
	return &b.buffer
}

// Unlocks for reading, to be called after Read.
func (b *Synced) ReadDone() {
	b.lock.RUnlock()
}

// Locks for writing and returns the (now locked) data Slice.
func (b *Synced) Write() *Slice {
	b.lock.Lock()
	return &b.buffer
}

// Unlocks for writing, to be called after Write.
func (b *Synced) WriteDone() {
	b.lock.Unlock()
}
