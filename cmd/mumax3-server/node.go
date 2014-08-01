package main

import (
	"log"
	"net/rpc"
	"reflect"
	"runtime"
)

// slave runs compute jobs
type node struct {
	Info
	pipe  chan func()
	peers map[string]*Conn
}

func (s *node) Init(addr string) {
	s.pipe = make(chan func(), 10)
	s.peers = make(map[string]*Conn)
	s.Addr = addr
}

func (n *node) mainLoop() {
	for {
		log.Println("waiting")
		f := <-n.pipe
		log.Println("working", runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name())
		f()
	}
}

// slave info sent out to masters
type Info struct {
	Addr string
}

// RPC client for a master
type Conn struct {
	Info
	*rpc.Client
}

// Public RPC service
type Node node

// Called by a peer to connect.
// MasterInfo will be stored by slave, SlaveInfo is stored by master.
func (N *Node) Connect(caller Info, myInfo *Info) error {
	n := (*node)(N)
	n.addPeer(caller)
	*myInfo = n.Info
	return nil
}

func (s *node) addPeer(peer Info) {
	s.pipe <- func() {
		if _, ok := s.peers[peer.Addr]; !ok {
			s.peers[peer.Addr] = &Conn{Info: peer}
		}
	}
}

//
//
//func DiscoverServers(baseIP, port string) {
//	log.Println("Scanning for servers")
//	for i := 1; i < 255; i++ {
//		ExploreServer(fmt.Sprint(baseIP, i, port))
//	}
//}
//
//func ExploreServer(addr string) {
//	client, err := rpc.DialHTTP("tcp", addr)
//	if err != nil {
//		return // nothing found
//	}
//
//	var reply struct{}
//	err = client.Call("RPC.Ping", struct{}{}, &reply)
//	if err == nil {
//		pipeline <- func() { AddServer(addr) }
//	} else {
//		log.Println(addr, ":", err)
//	}
//
//}
//
//type RPC struct{}
//
//
//func StartWorkers(n int) {
//	for i := 0; i < n; i++ {
//		go func() {
//			for {
//				(<-workers)()
//			}
//		}()
//	}
//}
