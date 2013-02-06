package mx

/*
   To my future colleagues:
   This code is the very core, the heart and soul of all concurrent GPU-CPU logic.
   Please, take a deep breath before editing. Raptors will kill you if break it.
   -Arne.
*/
import (
	"sync"
)

type mutex interface {
	lockNext(delta int)
	unlock()
	lockedRange() (start, stop int)
	isLocked() bool
}

// RWMutex protects an array for safe access by
// one writer and many readers.
// RWMutex makes sure the readers receive all data
// exactly once and in the correct order.
// Note that it is safe for the writer to also read
// the data (when he holds the write lock).
type rwMutex struct {
	cond       sync.Cond  // wait condition: read/write is safe
	state      sync.Mutex // protects the internal state, used in cond.
	absA, absB int64      // half-open interval locked for writing, moves indefinitely upwards.
	n          int        // Total number of elements in protected array.
	readers    []*rMutex  // all readers who can access this rwmutex
}

// RWMutex to protect an array of length N.
func newRWMutex(N int) *rwMutex {
	m := new(rwMutex)
	m.n = N
	m.cond = *(sync.NewCond(&m.state))
	return m
}

// Move the locked window
//func (m *rwMutex) delta(Δstart, Δstop int) {
//	m.cond.L.Lock()
//	m._delta(Δstart, Δstop)
//	m.cond.L.Unlock()
//	m.cond.Broadcast()
//}

// unsynchronized Delta
func (m *rwMutex) delta(Δstart, Δstop int) {
	Δa, Δb := int64(Δstart), int64(Δstop)
	rnge := int((m.absB + Δb) - (m.absA + Δa))
	if rnge < 0 || rnge > m.n || Δa < 0 || Δb < 0 {
		Panicf("rwmutex: delta out of range: Δstart=%v, Δstop=%v, N=%v", Δstart, Δstop, m.n)
	}
	for !m.canWLock(m.absA+Δa, m.absB+Δb) {
		m.cond.Wait()
	}
	m.absA += Δa
	m.absB += Δb
}

// Lock the next delta elements.
func (m *rwMutex) lockNext(delta int) {
	m.cond.L.Lock()
	if m.absA != m.absB {
		panic("rwmutex: lock of locked mutex")
	}
	m.delta(0, delta)
	m.cond.L.Unlock()
	m.cond.Broadcast()
}

// Unlock the previous interval locked for writing.
func (m *rwMutex) unlock() {
	m.cond.L.Lock()
	if m.absA == m.absB {
		panic("rwmutex: unlock of unlocked mutex")
	}
	rnge := int(m.absB - m.absA)
	m.delta(rnge, 0)
	m.cond.L.Unlock()
	m.cond.Broadcast()
}

// Returns the currently locked range.
// It is not thread-safe because the rwmutex is only
// supposed to be accessed by one writer thread.
func (m *rwMutex) lockedRange() (start, stop int) {
	return int(m.absA % int64(m.n)), int((m.absB-1)%int64(m.n)) + 1
}

// Can m safely lock for writing [a, b[ ?
func (m *rwMutex) canWLock(a, b int64) (ok bool) {
	// Panic if there are no readers.
	// This is definitely a mistake, as no readers may be added later.
	if len(m.readers) == 0 {
		panic("rwmutex: write without readers")
	}
	for _, r := range m.readers {
		if a < r.absD || b > (r.absC+int64(m.n)) {
			return false
		}
	}
	return true
}

func (m *rwMutex) isLocked() bool {
	want := m.absA
	if m.absB != want {
		return true
	}
	for i := range m.readers {
		if m.readers[i].absC != want || m.readers[i].absD != want {
			return true
		}
	}
	return false
}

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
//func (m *rMutex) ReadDelta(Δstart, Δstop int) {
//	m.rw.cond.L.Lock()
//	m.delta(Δstart, Δstop)
//	m.rw.cond.L.Unlock()
//	m.rw.cond.Broadcast()
//}

// unsynchronized Delta
func (m *rMutex) delta(Δstart, Δstop int) {
	Δc, Δd := int64(Δstart), int64(Δstop)
	rnge := int((m.absD + Δd) - (m.absC + Δc))
	if rnge < 0 || rnge > m.rw.n || Δc < 0 || Δd < 0 {
		Panicf("rwmutex: delta out of range: Δstart=%v, Δstop=%v, N=%v", Δstart, Δstop, m.rw.n)
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

// returns the currently read-locked range.
// It is not thread-safe because each RMutex is only
// supposed to be accessed by one reader thread.
func (m *rMutex) lockedRange() (start, stop int) {
	return int(m.absC % int64(m.rw.n)), int((m.absD-1)%int64(m.rw.n)) + 1
}

// Can m safely lock for reading [start, stop[ ?
func (r *rMutex) canRLock(c, d int64) (ok bool) {
	return r.rw.absA >= d && r.rw.absB <= (c+int64(r.rw.n))
}

func (m *rMutex) isLocked() bool {
	return m.rw.isLocked()
}
