package core

import (
	"sync"
)

type RWMutex struct {
	N         int // Total number of elements in protected array.
	a, b      int // half-open interval locked for writing
	timestamp int // time stamp of data before a, increments at each new frame.
	c, d      int // half-open interval locked for reading
	state     sync.Mutex
	cond      sync.Cond
}

func NewRWMutex(N int) *RWMutex {
	m := new(RWMutex)
	m.N = N
	m.cond = *(sync.NewCond(&m.state))
	return m
}

// Lock for writing [start, stop[.
func (m *RWMutex) Lock(start, stop int) {
	if start > stop || start >= m.N || stop > m.N || start < 0 || stop < 0 {
		Panicf("rwmutex: lock: invalid arguments: start=%v, stop=%v, n=%v", start, stop, m.N)
	}

	m.cond.L.Lock()
	for !m.canWLock(start, stop) {
		Debug("lock: wait")
		m.cond.Wait()
	}
	m.a, m.b = start, stop
	m.cond.L.Unlock()
	m.cond.Broadcast()
}

// Lock for reading [start, stop[.
func (m *RWMutex) RLock(start, stop int) {
	if start > stop || start >= m.N || stop > m.N || start < 0 || stop < 0 {
		Panicf("rwmutex: rlock: invalid arguments: start=%v, stop=%v, n=%v", start, stop, m.N)
	}
	m.cond.L.Lock()
	for !m.canRLock(start, stop) {
		Debug("rlock: wait")
		m.cond.Wait()
	}
	m.c, m.d = start, stop
	m.cond.L.Unlock()
	m.cond.Broadcast()
}

// Can m safely lock for writing [start, stop[ ?
// Not thread-safe, assumes state mutex is locked.
func (m *RWMutex) canWLock(a, b int) bool {
	c, d := m.c, m.d
	// intersection of read & write interval:
	ia, ib := max(a, c), min(b, d)
	Logf("canwlock: a=%v, b=%v, c=%v, d=%v", a, b, c, d)
	ok := (ib <= ia) // intersection should be empty
	Log("canwlock", a, b, ":", ok)
	return ok
}

// Can m safely lock for reading [start, stop[ ?
// Not thread-safe, assumes state mutex is locked.
func (m *RWMutex) canRLock(c, d int) bool {
	a, b := m.a, m.b
	// intersection of read & write interval:
	ia, ib := max(a, c), min(b, d)
	Logf("canrlock: a=%v, b=%v, c=%v, d=%v", a, b, c, d)
	ok := (ib <= ia) // intersection should be empty
	Log("canrlock", c, d, ":", ok)
	return ok
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// Time stamp when data at index has last been written.
// Not thread-safe, assumes state mutex is locked.
func (m *RWMutex) stampOf(index int) int {
	if index < m.a {
		return m.timestamp
	}
	if index >= m.b {
		return m.timestamp - 1
	}
	Panicf("rwmutex: timestamp: invalid index: start=%v, stop=%v, index=%v", m.a, m.b, index)
	return -2 // silence gc (dummy value)
}

//// Protects an array for thread-safe
//// concurrent writing by one writer
//// and reading by many readers.
//// The writer locks parts of the array with
//// 	Lock(start, stop)
//// 
//type rnge struct {
//	start, stop int
//}
//
//type RWMutex struct {
//	state sync.Mutex
//	N     int  // total number of elements in protected array
//	w, r  rnge // writer/reader locked range
//	//readers []RLock
//}
//
//// Locks for writing between indices 
//// start (inclusive) and stop (exclusive).
//func (m *RWMutex) Lock(start, stop int) {
//	// check bounds
//	if start > stop || start >= m.N || stop > m.N || start < 0 || stop < 0 {
//		panic(fmt.Errorf("rwmutex: lock: invalid arguments: start=%v, stop=%v, n=%v", start, stop, m.N))
//	}
//
//	m.state.Lock()
//	{
//		m.w.start = start
//		m.w.stop = stop
//	}
//	m.state.Unlock()
//}
//
//// Registers and returns a new lock for reading.
//func (m *RWMutex) MakeRLock() *RLock {
//	return nil
//}
//
//// Lock for reading a RWMutex.
//type RLock struct {
//}
//
//// Locks for reading between indices 
//// start (inclusive) and stop (exclusive).
//func (m *RLock) RLock(start, stop int) {
//
//}
