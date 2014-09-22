package main

type User struct {
	Jobs      []*Job
	FairShare float64 // Used-up compute time in the past (decays)
}

func NewUser() *User {
	return &User{}
}

//// used share in GPU*hours
//func (u *Share) Used() float64 {
//	used := u.used
//	for _, j := range u.Running {
//		used += j.Runtime().Seconds()
//	}
//	return used / 3600
//}

//func (u *User) HasJob() bool {
//	return len(u.Queue) > 0
//}
//
//func (u *User) GiveJob(nodeAddr string) string {
//	var job *Job
//
//	// take random job from Queue map
//	for _, j := range u.Queue {
//		job = j
//		break
//	}
//	delete(u.Queue, job.URL)
//
//	job.Status = RUNNING
//	job.Start = time.Now()
//	job.Node = nodeAddr
//	job.Node = nodeAddr
//
//	u.Running[job.URL] = job
//
//	//log.Println("give job", job.URL, "->", nodeAddr)
//	return job.URL
//}
