package core

import "sync"

// One reader, one writer.
type RWMutex struct {
	N            int        // Total number of elements in protected array.
	a, b         int        // half-open interval locked for writing
	writingframe int        // time stamp of data currently being written in [a, b[
	state        sync.Mutex // protects the internal state, used in cond.
	cond         sync.Cond  // wait condition: read/write is safe
	readers      []*RMutex  // all readers who can access this rwmutex
}

func NewRWMutex(N int) *RWMutex {
	m := new(RWMutex)
	m.N = N
	m.cond = *(sync.NewCond(&m.state))
	m.writingframe = -1 // nothing yet written
	return m
}

func (m *RWMutex) NewReader() *RMutex {
	m.cond.L.Lock()
	defer m.cond.L.Unlock()
	r := &RMutex{rw: m, c: 0, d: 0, lastread: -1}
	m.readers = append(m.readers, r)
	return r
}

type RMutex struct {
	rw       *RWMutex
	c, d     int // half-open interval locked for reading
	lastread int // time stamp of data last read in [c, d[
}

// Lock for reading [start, stop[.
// Automatically unlocks the previous interval.
// RLock(0, 0) can be used to explicitly unlock.
func (m *RMutex) RLock(start, stop int) {
	if start > stop || start >= m.rw.N || stop > m.rw.N || start < 0 || stop < 0 {
		Panicf("rwmutex: rlock: invalid arguments: start=%v, stop=%v, n=%v", start, stop, m.rw.N)
	}

	m.rw.cond.L.Lock()
	//Debug("RLock", start, stop)
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

// Lock for writing [start, stop[.
// Automatically unlocks the previous interval.
// Lock(0, 0) can be used to explicitly unlock.
func (m *RWMutex) Lock(start, stop int) {

	if start > stop || start >= m.N || stop > m.N || start < 0 || stop < 0 {
		Panicf("rwmutex: lock: invalid arguments: start=%v, stop=%v, n=%v", start, stop, m.N)
	}

	m.cond.L.Lock()
	//Debug("WLock", start, stop)
	if start == 0 {
		m.writingframe++
		//Debug("W new frame, writingframe=", m.writingframe)
	}
	m.a, m.b = start, start // noting is being written while waiting
	for !m.canWLock(start, stop) {
		//Debug("Wlock: wait")
		m.cond.Wait()
	}
	m.a, m.b = start, stop // update lock the interval
	m.cond.L.Unlock()
	m.cond.Broadcast()
}

// Can m safely lock for writing [start, stop[ ?
// Not thread-safe, assumes state mutex is locked.
func (m *RWMutex) canWLock(a, b int) (ok bool) {
	for _, r := range m.readers {
		c, d := r.c, r.d
		//reason := "?"
		//defer func() { Debug("canWlock: [", a, ",", b, "[, [", c, ",", d, "[", ok, reason) }()

		// intersection of read & write interval:
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

// ______________________________________________________ Read

// Can m safely lock for reading [start, stop[ ?
// Not thread-safe, assumes state mutex is locked.
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
// Not thread-safe, assumes state mutex is locked.
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
