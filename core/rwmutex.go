package core

/* 
   To my future colleagues:
   This code is the very core, the heart and soul of all concurrent GPU-CPU logic.
   Please, take a deep breath before editing. Raptors will kill you if break it.
   -Arne.
*/

import "sync"

// RWMutex protects an array for safe access by
// one writer and many readers. 
// Example:
// 	array := make([]whatever, 100)
// 	wlock := NewRWMutex(len(array))
// 	rlock := wlock.NewReader()
// 	
// 	go func(){
// 		wlock.WLock(0, 50)
// 		// write elements 0 to 50
// 		wlock.WLock(50, 100)
// 		// write elements 50 to 100
// 		wlock.WLock(0, 50)
// 		// ...
// 	}()
// 	
// 	rlock.RLock(0, 20)
// 	// read elemnts 0 to 20
// 	// ...
// RWMutex makes sure the readers receive all data
// exactly once and in the correct order.
type RWMutex struct {
	N            int        // Total number of elements in protected array.
	a, b         int        // half-open interval locked for writing
	writingframe int        // time stamp of data currently being written in [a, b[
	state        sync.Mutex // protects the internal state, used in cond.
	cond         sync.Cond  // wait condition: read/write is safe
	readers      []*RMutex  // all readers who can access this rwmutex
}

// RWMutex to protect an array of length N.
func NewRWMutex(N int) *RWMutex {
	m := new(RWMutex)
	m.N = N
	m.cond = *(sync.NewCond(&m.state))
	m.writingframe = 0 // frame 0 is next one to be written
	return m
}

// Make a new read lock for this RWMutex.
func (m *RWMutex) NewReader() *RMutex {
	m.cond.L.Lock()
	defer m.cond.L.Unlock()
	r := &RMutex{rw: m, c: 0, d: 0, lastread: -1}
	m.readers = append(m.readers, r)
	return r
}

// RMutex is a read-only lock, created by an RWMutex.
type RMutex struct {
	rw       *RWMutex
	c, d     int // half-open interval locked for reading
	lastread int // time stamp of data last read in [c, d[
}

// Lock for reading [start, stop[.
// Automatically unlocks the previous interval.
// RLock(0, 0) can be used to explicitly unlock.
func (m *RMutex) RLock(start, stop int) {
	m.rw.check(start, stop)
	m.rw.cond.L.Lock()
	m.c, m.d = start, start
	for !m.canRLock(start, stop) {
		//Debug("Rlock: wait")
		m.rw.cond.Wait()
	}
	m.c, m.d = start, stop
	if stop == m.rw.N {
		m.lastread++
		//Debug("R new frame, lastread=", m.lastread)
	}
	m.rw.cond.L.Unlock()
	m.rw.cond.Broadcast() // TODO: benchmark with broadcast in/out lock.
}

// TryRLock tries to acquire the lock without blocking.
// If the lock cannot be acquired, TryRLock() 
// immediately returns false.
// Otherwise the lock is acquired, returning true.
func (m *RMutex) TryRLock(start, stop int) (ok bool) {
	m.rw.check(start, stop)
	m.rw.cond.L.Lock()
	m.c, m.d = start, start
	if !m.canRLock(start, stop) {
		ok = false
		m.rw.cond.L.Unlock()
		m.rw.cond.Broadcast()
		return
	}
	m.c, m.d = start, stop
	if stop == m.rw.N {
		m.lastread++
	}
	ok = true
	m.rw.cond.L.Unlock()
	m.rw.cond.Broadcast()
	return
}

// Lock for writing [start, stop[.
// Automatically unlocks the previous interval.
// Lock(0, 0) can be used to explicitly unlock.
func (m *RWMutex) WLock(start, stop int) {
	m.check(start, stop)
	m.cond.L.Lock()
	m.a, m.b = start, start // noting is being written while waiting
	for !m.canWLock(start, stop) {
		m.cond.Wait()
	}
	m.a, m.b = start, stop // update lock the interval
	if stop == m.N {
		m.writingframe++
	}
	m.cond.L.Unlock()
	m.cond.Broadcast()
}

// TryWLock tries to acquire the lock without blocking.
// If the lock cannot be acquired, TryWLock() 
// immediately returns false.
// Otherwise the lock is acquired, returning true.
func (m *RWMutex) TryWLock(start, stop int) (ok bool) {
	m.check(start, stop)
	m.cond.L.Lock()
	m.a, m.b = start, start // noting is being written while waiting
	if !m.canWLock(start, stop) {
		ok = false
		m.cond.L.Unlock()
		m.cond.Broadcast() // previous interval is now unlocked, so broadcast
		return
	}
	m.a, m.b = start, stop // update lock the interval
	if stop == m.N {
		m.writingframe++
	}
	ok = true
	m.cond.L.Unlock()
	m.cond.Broadcast()
	return
}

// Can m safely lock for writing [start, stop[ ?
func (m *RWMutex) canWLock(a, b int) (ok bool) {
	for _, r := range m.readers {
		c, d := r.c, r.d
		//reason := "?"
		//defer func() { Debug("canWlock: [", a, ",", b, "[, [", c, ",", d, "[", ok, reason) }()
		ok = !intersects(a, b, c, d)
		if !ok {
			//reason = "intersects"
			return
		}
		// make sure we don't overwrite data that has not yet been read.
		if a >= d {
			if m.stampOf(a) != r.lastread { // time stamp should be OK
				//reason = fmt.Sprint("stampOf", a, "==", m.stampOf(a), "!=", m.lastread)
				return false
			}
		}
	}
	//reason = "all ok"
	return true
}

// Can m safely lock for reading [start, stop[ ?
func (r *RMutex) canRLock(c, d int) (ok bool) {
	m := r.rw
	a, b := m.a, m.b
	//reason := "?"
	//defer func() { Debug("canRlock: [", a, ",", b, "[, [", c, ",", d, "[", ok, reason) }()
	ok = !intersects(a, b, c, d) // intersection should be empty
	if !ok {
		//reason = "intersects"
		return
	}
	// make sure we don't read data that has not yet been written.
	if c >= b {
		if m.stampOf(d) != r.lastread+1 { // time stamp should be OK
			//reason = fmt.Sprint("stampOf", d, "==", m.stampOf(d), "!=", m.lastread, "+ 1")
			ok = false
			return
		}
	}
	//reason = "ok"
	return true
}

// [a, b[ intersects [c, d[ ?
func intersects(a, b, c, d int) bool {
	return max(a, c) < min(b, d)
}

// Time stamp when data at index has last been written.
func (m *RWMutex) stampOf(index int) int {
	if index < m.a {
		return m.writingframe
	}
	if index >= m.b {
		return m.writingframe - 1
	}
	Panicf("rwmutex: writingframe: invalid index: start=%v, stop=%v, index=%v", m.a, m.b, index)
	return -2 // silence gc (dummy value)
}

func (m *RWMutex) check(start, stop int) {
	if start > stop || start >= m.N || stop > m.N || start < 0 || stop < 0 {
		Panicf("rwmutex: lock: invalid arguments: start=%v, stop=%v, n=%v", start, stop, m.N)
	}
}
