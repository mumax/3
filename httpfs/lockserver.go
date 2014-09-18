package httpfs

import (
	"sync"
	"time"
)

type LockServer struct {
	leases        map[string]*lease
	capacity      int
	mutex         sync.Mutex
	leaseDuration time.Duration
}

func NewLockServer(leaseDuration time.Duration) *LockServer {
	return &LockServer{leases: make(map[string]*lease), capacity: 1, leaseDuration: leaseDuration}
	// Pike suggests capacity 1 to ensure re-size code is exercised.
}

type lease struct {
	owner string
	t     time.Time
}

func (l *LockServer) Lock(owner, key string) (ok bool) {
	l.mutex.Lock()
	defer l.mutex.Unlock()

	now := time.Now()

	if lse, ok := l.leases[key]; ok {
		// already own lease or lease expired: (re-)acquire
		if lse.owner == owner || now.Sub(lse.t) > l.leaseDuration {
			lse.t = now
			lse.owner = owner
			return true
		} else { // don't own and not yet expired
			return false
		}
	} else {
		// no such lease yet
		l.cleanup()
		l.leases[key] = &lease{owner: owner, t: now}
		return true
	}
}

// remove expired leases. a run is linear in the total number of leases,
// so we amortize by keeping a "capacity", above which we clean up and
// then double the capacity.
func (l *LockServer) cleanup() {
	if len(l.leases) < l.capacity {
		return
	}

	now := time.Now()

	for k, lse := range l.leases {
		if now.Sub(lse.t) > l.leaseDuration {
			delete(l.leases, k)
		}
	}

	// increase even if removing expired keys culled the number of leases entries below capacity,
	// avoids expensive cleanup when we continuously add/remove around the capacity border.
	l.capacity *= 2
}
