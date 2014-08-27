package main

import (
	"log"
	"reflect"
	//"runtime"
	"sync"
	"time"
)

type Node struct {
	Addr         string // canonical (unique) address of node, read-only
	RootDir      string // httpfs storage root
	MumaxVersion string
	upSince      time.Time

	// compute service
	GPUs        []GPU
	RunningHere map[string]*Job

	Peers map[string]PeerInfo

	Users map[string]*User

	mutex sync.Mutex
	value reflect.Value
	lockT time.Time
}

func (n *Node) Uptime() time.Duration {
	return since(time.Now(), n.upSince)
}

func since(a, b time.Time) time.Duration {
	d := a.Sub(b)
	return (d / 1e9) * 1e9
}

func (n *Node) lock() {
	//	pc, file, line, _ := runtime.Caller(1)
	//	log.Println(" -> wait ", runtime.FuncForPC(pc).Name(), file, line)
	n.mutex.Lock()
	//log.Println(" -> lock ", runtime.FuncForPC(pc).Name(), file, line)
	n.lockT = time.Now()
}

const maxLatency = 1 * time.Second

func (n *Node) unlock() {
	d := time.Since(n.lockT)
	if d > maxLatency {
		log.Println("*** locked for more than", d)
	}
	//pc, file, line, _ := runtime.Caller(1)
	//log.Println(" <- unlock ", runtime.FuncForPC(pc).Name(), file, line, "after", d)
	n.mutex.Unlock()
}
