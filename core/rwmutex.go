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
type RWMutex struct {
	n          int        // Total number of elements in protected array.
	absA, absB int64      // half-open interval locked for writing
	state      sync.Mutex // protects the internal state, used in cond.
	cond       sync.Cond  // wait condition: read/write is safe
	readers    []*RMutex  // all readers who can access this rwmutex
}

// RWMutex to protect an array of length N.
func NewRWMutex(N int) *RWMutex {
	m := new(RWMutex)
	m.n = N
	m.cond = *(sync.NewCond(&m.state))
	return m
}

// Make a new read lock for this RWMutex.
func (m *RWMutex) NewReader() *RMutex {
	m.cond.L.Lock()
	defer m.cond.L.Unlock()
	r := &RMutex{rw: m}
	m.readers = append(m.readers, r)
	return r
}

// RMutex is a read-only lock, created by an RWMutex.
type RMutex struct {
	rw         *RWMutex
	absC, absD int64 // half-open interval locked for reading
}

// Lock for reading [start, stop[.
// Automatically unlocks the previous interval.
// RLock(0, 0) can be used to explicitly unlock.
func (m *RMutex) RLock(delta int) {
	m.rw.cond.L.Lock()

	delta64 := int64(delta)
	if m.absC != m.absD {
		panic("rmutex: lock of locked mutex")
	}
	for !m.canRLock(m.absC, m.absC+delta64) {
		m.rw.cond.Wait()
	}
	m.absD = m.absC + delta64

	m.rw.cond.L.Unlock()
	m.rw.cond.Broadcast()
}

func (m *RMutex) RUnlock() {
	m.rw.cond.L.Lock()

	m.absC = m.absD

	m.rw.cond.L.Unlock()
	m.rw.cond.Broadcast()
}

func (m *RWMutex) WLock(delta int) {
	// TODO: test delta, rnge >= 0 and <= N
	m.cond.L.Lock()

	delta64 := int64(delta)
	if m.absA != m.absB {
		panic("rwmutex: lock of locked mutex")
	}
	for !m.canWLock(m.absA, m.absA+delta64) {
		m.cond.Wait()
	}
	m.absB = m.absA + delta64

	m.cond.L.Unlock()
	m.cond.Broadcast()
}

func (m *RWMutex) WUnlock() {
	m.cond.L.Lock()
	if m.absA == m.absB {
		panic("rwmutex: unlock of unlocked mutex")
	}
	m.absA = m.absB

	m.cond.L.Unlock()
	m.cond.Broadcast()
}

// Can m safely lock for writing [start, stop[ ?
func (m *RWMutex) canWLock(a, b int64) (ok bool) {
	for _, r := range m.readers {
		if a < r.absD || b > (r.absC+int64(m.n)) {
			Debug("canWLock", a, b, ": false")
			return false
		}
	}
	Debug("canWLock", a, b, ": true")
	return true
}

// Can m safely lock for reading [start, stop[ ?
func (r *RMutex) canRLock(c, d int64) (ok bool) {
	ok = r.rw.absA >= d && r.rw.absB <= (c+int64(r.rw.n))
	Debug("canRLock", c, d, ok)
	return ok
}

//// TryWLock tries to acquire the lock without blocking.
//// If the lock cannot be acquired, TryWLock() 
//// immediately returns false.
//// Otherwise the lock is acquired, returning true.
//func (m *RWMutex) TryWLock(start, stop int) (ok bool) {
//	m.check(start, stop)
//	m.cond.L.Lock()
//	m.a, m.b = start, start // noting is being written while waiting
//	if !m.canWLock(start, stop) {
//		ok = false
//		m.cond.L.Unlock()
//		m.cond.Broadcast() // previous interval is now unlocked, so broadcast
//		return
//	}
//	m.a, m.b = start, stop // update lock the interval
//	if stop == m.N {
//		m.writingframe++
//	}
//	ok = true
//	m.cond.L.Unlock()
//	m.cond.Broadcast()
//	return
//}

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
