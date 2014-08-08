package main

import (
	"fmt"
	"log"
)

const (
	N_SCANNERS = 8
	MaxIPs     = 1024
)

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
	ret, err := node.RPCCall(addr, "Ping", node.Info())
	if err == nil {
		peerInfo := ret.(NodeInfo)
		node.AddPeer(peerInfo)
	} else {
		//log.Println("probe", addr, ":", err)
	}
}
