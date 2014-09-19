package main

//
//import (
//	"log"
//	"reflect"
//	"sync"
//	"time"
//)
//
//type Node struct {
//	Addr         string // canonical (unique) address of node, read-only
//
//	// compute service
//
//	Peers map[string]PeerInfo
//
//	LastJobScanTime  time.Time // last time job scanner ran (for status reporting)
//	LastJobScanFiles int       // how many new jobs scanner picked up last time (for status)
//
//	mutex sync.Mutex
//	value reflect.Value
//	lockT time.Time
//}
//
//
//func (n *Node) lock() {
//	//	pc, file, line, _ := runtime.Caller(1)
//	//	log.Println(" -> wait ", runtime.FuncForPC(pc).Name(), file, line)
//	n.mutex.Lock()
//	//log.Println(" -> lock ", runtime.FuncForPC(pc).Name(), file, line)
//	n.lockT = time.Now()
//}
//
//const maxLatency = 1 * time.Second
//
//func (n *Node) unlock() {
//	d := time.Since(n.lockT)
//	if d > maxLatency {
//		log.Println("*** locked for more than", d)
//	}
//	//pc, file, line, _ := runtime.Caller(1)
//	//log.Println(" <- unlock ", runtime.FuncForPC(pc).Name(), file, line, "after", d)
//	n.mutex.Unlock()
//}
