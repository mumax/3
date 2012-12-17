package nimble

import (
	"code.google.com/p/mx3/core"
)

// RMutex is a read-only lock, created by an RWMutex.
type rMutex struct {
	rw         *rwMutex
	absC, absD int64 // half-open interval locked for reading
}

// Make a new read lock for this RWMutex.
func (m *rwMutex) MakeRMutex() *rMutex {
	m.cond.L.Lock()
	defer m.cond.L.Unlock()
	// Panic if we are already writing to the RWMutex.
	if m.absB > 0 {
		panic("rwmutex already in use")
	}
	r := &rMutex{rw: m}
	m.readers = append(m.readers, r)
	return r
}

// Move the locked window
func (m *rMutex) ReadDelta(Δstart, Δstop int) {
	m.rw.cond.L.Lock()
	m.delta(Δstart, Δstop)
	m.rw.cond.L.Unlock()
	m.rw.cond.Broadcast()
}

// unsynchronized Delta
func (m *rMutex) delta(Δstart, Δstop int) {
	Δc, Δd := int64(Δstart), int64(Δstop)
	rnge := int((m.absD + Δd) - (m.absC + Δc))
	if rnge < 0 || rnge > m.rw.n || Δc < 0 || Δd < 0 {
		core.Panicf("rwmutex: delta out of range: Δstart=%v, Δstop=%v, N=%v", Δstart, Δstop, m.rw.n)
	}
	for !m.canRLock(m.absC+Δc, m.absD+Δd) {
		m.rw.cond.Wait()
	}
	m.absC += Δc
	m.absD += Δd
}

// Lock the next delta elements for reading.
func (m *rMutex) lockNext(delta int) {
	m.rw.cond.L.Lock()
	if m.absC != m.absD {
		panic("rmutex: readnext w/o readdone")
	}
	m.delta(0, delta)
	m.rw.cond.L.Unlock()
	m.rw.cond.Broadcast()
}

// Unlock the previous interval locked for reading.
func (m *rMutex) unlock() {
	m.rw.cond.L.Lock()
	rnge := int(m.absD - m.absC)
	m.delta(rnge, 0)
	m.rw.cond.L.Unlock()
	m.rw.cond.Broadcast()
}

// RRange returns the currently read-locked range.
// It is not thread-safe because each RMutex is only
// supposed to be accessed by one reader thread.
func (m *rMutex) lockedRange() (start, stop int) {
	return int(m.absC % int64(m.rw.n)), int((m.absD-1)%int64(m.rw.n)) + 1
}

// Can m safely lock for reading [start, stop[ ?
func (r *rMutex) canRLock(c, d int64) (ok bool) {
	return r.rw.absA >= d && r.rw.absB <= (c+int64(r.rw.n))
}
