package main

import (
	"log"
	"reflect"
	"sync"
	"time"
)

type Node struct {
	inf           NodeInfo
	MumaxVersion  string
	GPUs          []GPU
	upSince       time.Time
	peers         map[string]NodeInfo
	m             sync.Mutex
	jobs, running []Job
	value         reflect.Value
	lockT         time.Time
}

type NodeInfo struct {
	Addr string // node's RPC address, also serves as unique ID
}

func (n *Node) Uptime() time.Duration {
	return time.Since(n.upSince)
}

// Thread-safe info()
func (n *Node) Info() NodeInfo {
	n.lock()
	defer n.unlock()
	return n.info()
}

func (n *Node) info() NodeInfo {
	return n.inf
}

// Thread-safe addPeer()
func (n *Node) AddPeer(pInf NodeInfo) {
	n.lock()
	defer n.unlock()
	n.addPeer(pInf)
}

func (n *Node) addPeer(pInf NodeInfo) {
	if n.peers == nil {
		n.peers = make(map[string]NodeInfo)
	}
	a := pInf.Addr
	if _, ok := n.peers[a]; !ok {
		log.Println("add new peer:", a)
		n.peers[a] = pInf
	}
}

// Thread-safe n.peers[pAddr]
func (n *Node) PeerInfo(pAddr string) (p NodeInfo, ok bool) {
	n.lock()
	defer n.unlock()
	p, ok = n.peers[pAddr]
	return
}

func (n *Node) lock() {
	n.m.Lock()
	n.lockT = time.Now()
}

const maxLatency = 1 * time.Second

func (n *Node) unlock() {
	d := time.Since(n.lockT)
	if d > maxLatency {
		log.Println("*** locked for more than", d)
	}
	n.m.Unlock()
}
