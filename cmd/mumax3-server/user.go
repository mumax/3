package main

import "time"

type User struct {
	Jobs      []*Job
	FairShare float64 // Used-up compute time in the past (decays)
	nextPtr   int     // pointer suggesting next job to start. Reset on re-scan. len(Jobs) means no queued job
}

func NewUser() *User {
	return &User{}
}

// nextJob looks for the next free job in the list.
// it does a tiny bit of linear search, starting from nextPtr.
func (u *User) giveJob(node string) *Job {
	index := u.nextJobPtr()
	if index >= len(u.Jobs) {
		return nil
	}
	u.nextPtr++
	j := u.Jobs[index]
	// all below are preliminary, to get rapid gui response.
	// may be overwritten by update
	j.Host = node
	j.Output = OutputDir(j.ID)
	j.Start = time.Now()
	return j
}

func (u *User) HasJob() bool {
	i := u.nextJobPtr()
	return i < len(u.Jobs)
}

// returns
func (u *User) nextJobPtr() int {
	for ; u.nextPtr < len(u.Jobs); u.nextPtr++ {
		j := u.Jobs[u.nextPtr]
		if j.IsQueued() {
			return u.nextPtr
		}
	}
	return u.nextPtr
}
