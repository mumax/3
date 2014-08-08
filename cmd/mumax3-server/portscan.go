package main

import (
	"fmt"
	"log"
	"net"
	"net/rpc"
)

const (
	N_SCANNERS = 128
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
	conn, err := net.DialTimeout("tcp", addr, *flag_timeout)
	if err != nil {
		return
	}

	client := rpc.NewClient(conn)
	defer client.Close()

	var pInf NodeInfo
	err = client.Call("RPC.Ping", node.Info(), &pInf)
	if err == nil {
		node.AddPeer(pInf)
	} else {
		log.Println("probe", addr, ":", err)
	}
}
