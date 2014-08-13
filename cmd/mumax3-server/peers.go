package main

// Peer management:
//  portscan for peers
// 	ping peers

import (
	"fmt"
	"log"
)

const (
	N_SCANNERS = 8
	MaxIPs     = 1024
)

type PeerInfo struct {
	Addr     string
	HaveJobs bool
}

func (n *Node) AddPeer(pAddr string) {
	n.lock()
	defer n.unlock()
	if n.peers == nil {
		n.peers = make(map[string]PeerInfo)
	}
	if _, ok := n.peers[pAddr]; !ok {
		log.Println("add new peer:", pAddr)
		n.peers[pAddr] = PeerInfo{
			Addr:     pAddr,
			HaveJobs: true, // makes sure we ask for jobs at least once
		}
	}
}

// Thread-safe n.peers[pAddr]
func (n *Node) PeerInfo(pAddr string) (p PeerInfo, ok bool) {
	n.lock()
	defer n.unlock()
	p, ok = n.peers[pAddr]
	return
}

func FindPeers(IPs []string, minPort, maxPort int) {
	log.Println("Portscan start")

	scanners := make(chan func())

	for i := 0; i < N_SCANNERS; i++ {
		go func() {
			for f := range scanners {
				f()
			}
		}()
	}

	for _, ip := range IPs {
		for port := minPort; port <= maxPort; port++ {
			addr := fmt.Sprint(ip, ":", port)
			scanners <- func() { ProbePeer(addr) }
		}
	}
	close(scanners)
	log.Println("Portscan done")
}

//
func ProbePeer(addr string) {
	ret, err := node.RPCCall(addr, "Ping", node.Addr)
	if err == nil {
		peerAddr := ret.(string)
		node.AddPeer(peerAddr)
	} else {
		//log.Println("probe", addr, ":", err)
	}
}

// RPC-called
func (n *Node) Ping(peerAddr string) string {
	// Somebody just called my status,
	// and him as a peer (if not yet so).
	if _, ok := n.PeerInfo(peerAddr); !ok {
		n.AddPeer(peerAddr)
	}
	return n.Addr
}
