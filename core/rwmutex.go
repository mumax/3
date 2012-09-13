package core

import (
	"fmt"
)

// Protects an array for thread-safe
// concurrent writing by one writer
// and reading by many readers.
// The writer locks parts of the array with
// 	Lock(start, stop)
// 
type writer struct {
	start, stop int
	stamp       int
}

type reader writer

type RWMutex struct {
	N       int
	w       writer
	readers []RLock
}

// Locks for writing between indices 
// start (inclusive) and stop (exclusive).
func (m *RWMutex) Lock(start, stop int) {
	// check bounds
	if start > stop || start >= m.N || stop > m.N {
		panic(fmt.Errorf("rwmutex: lock: invalid arguments: start=%v, stop=%v, n=%v", start, stop, m.N))
	}

	// start new frame?
	if (start < m.w.start || stop < m.w.stop) && (start != 0 || stop != 0) {
		panic(fmt.Errorf("rwmutex: new frame should Lock(0, 0) got: start=%v, stop=%v", start, stop))
	}
	newframe := (start == 0 && stop == 0)

	// contiguous?
	if !newframe && m.w.stop != start {
		panic(fmt.Errorf("rwmutex: lock not contiguous: start=%v, previous stop=%v", start, m.w.stop))
	}

	m.w.start = start
	m.w.stop = stop
	if newframe {
		m.w.stamp++
	}
}

// Registers and returns a new lock for reading.
func (m *RWMutex) MakeRLock() *RLock {
	return nil
}

// Lock for reading a RWMutex.
type RLock struct {
}

// Locks for reading between indices 
// start (inclusive) and stop (exclusive).
func (m *RLock) RLock(start, stop int) {

}
