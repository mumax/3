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
func (s *Synced) Read() *Slice {
	s.lock.RLock()
	return &s.buffer
}

// Unlocks for reading, to be called after Read.
func (s *Synced) ReadDone() {
	s.lock.RUnlock()
}

// Locks for writing and returns the (now locked) data Slice.
func (s *Synced) Write() *Slice {
	s.lock.Lock()
	return &s.buffer
}

// Unlocks for writing, to be called after Write.
func (s *Synced) WriteDone() {
	s.lock.Unlock()
}

func (s *Synced) Mesh() *Mesh {
	return s.buffer.Mesh()
}
