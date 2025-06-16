package main

// Peer management:
//  portscan for peers
// 	ping peers

import (
	"fmt"
	"log"
)

var (
	peers = make(map[string]*Peer)
)

type Peer struct {
}

func AddPeer(pAddr string) {
	WLock()
	defer WUnlock()

	if _, ok := peers[pAddr]; !ok {
		log.Println("add new peer:", pAddr)
		peers[pAddr] = NewPeer()
	}
}

func NewPeer() *Peer {
	return &Peer{}
}

// RPC-called
func Ping(peerAddr string) string {
	WLock()
	defer WUnlock()

	// Somebody just called my status,
	// add him as a peer (if not yet so).
	if _, ok := peers[peerAddr]; !ok {
		peers[peerAddr] = NewPeer()
	}
	return thisAddr
}

// Ping peer at address, add to peers list if he responds and is not yet added
func ProbePeer(addr string) {
	ret, _ := RPCCall(addr, "Ping", thisAddr)
	if ret != "" {
		AddPeer(ret)
	}
}

// Scan IPs and port range for peers that respond to Ping,
// add them to peers list.
func FindPeers(IPs []string, minPort, maxPort int) {
	//log.Println("Portscan start")

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
	log.Println("-- portscan done")
}
