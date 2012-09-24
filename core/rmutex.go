package core

// RMutex is a read-only lock, created by an RWMutex.
type RMutex struct {
	rw         *RWMutex
	absC, absD int64 // half-open interval locked for reading
}

// Make a new read lock for this RWMutex.
func (m *RWMutex) NewReader() *RMutex {
	m.cond.L.Lock()
	defer m.cond.L.Unlock()
	r := &RMutex{rw: m}
	m.readers = append(m.readers, r)
	return r
}

// Lock the next delta elements for reading.
func (m *RMutex) ReadNext(delta int) {
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

// Unlock the previous interval locked for reading.
func (m *RMutex) ReadDone() {
	m.rw.cond.L.Lock()

	m.absC = m.absD

	m.rw.cond.L.Unlock()
	m.rw.cond.Broadcast()
}

// RRange returns the currently read-locked range.
// It is not thread-safe because each RMutex is only
// supposed to be accessed by one reader thread.
func (m *RMutex) RRange() (start, stop int) {
	return int(m.absC % int64(m.rw.n)), int(m.absD % int64(m.rw.n))
}

// Can m safely lock for reading [start, stop[ ?
func (r *RMutex) canRLock(c, d int64) (ok bool) {
	ok = r.rw.absA >= d && r.rw.absB <= (c+int64(r.rw.n))
	Debug("canRLock", c, d, ok)
	return ok
}
