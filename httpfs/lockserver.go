package httpfs

import (
	"sync"
	"time"
)

// A LockServer maps keys to a single owner for a certain lease time.
// The owner can indefinitely extend his lease by periodically repeating
// the Lock call. During that time no other owner can acquire the lock.
// When no longer needed, the owner just lets his lease expire,
// after which others can lock it.
type LockServer struct {
	leases        map[string]*lease // maps key -> {owner, time}
	gcLimit       int               // garbage collect when len(leases) > gcLimit
	leaseDuration time.Duration
	mutex         sync.Mutex
}

func NewLockServer(leaseDuration time.Duration) *LockServer {
	return &LockServer{leases: make(map[string]*lease), gcLimit: 1, leaseDuration: leaseDuration}
	// initial gcLimit=1 to ensure re-size code is exercised.
}

type lease struct {
	owner string
	t     time.Time
}

// Lock tries to acquire the key for owner and returns true in case of success.
// After successful acquisition, repeated calls within the lease time will
// continue to return true and no other owners can lock that key.
func (l *LockServer) Lock(owner, key string) (ok bool) {
	l.mutex.Lock()
	defer l.mutex.Unlock()

	now := time.Now()

	if len(l.leases) > l.gcLimit {
		l.gc()
	}

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
		l.leases[key] = &lease{owner: owner, t: now}
		return true
	}

}

// remove expired leases. a run is linear in the total number of leases,
// so we amortize by keeping a "capacity", above which we clean up and
// then double the capacity.
func (l *LockServer) gc() {
	now := time.Now()
	for k, lse := range l.leases {
		if now.Sub(lse.t) > l.leaseDuration {
			delete(l.leases, k)
		}
	}

	l.gcLimit = len(l.leases)*2 + 1 // +1 avoids zero
}
