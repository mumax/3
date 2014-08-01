package main

import (
	"fmt"
	"log"
	"net"
	"net/rpc"
)

const N_SCANNERS = 128

func FindPeers(IPs []string, minPort, maxPort int, myStatus Status) {
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
			scanners <- func() { RPCProbe(addr, myStatus) }
		}
	}
	close(scanners)
	log.Println("Portscan done")
}

func RPCProbe(addr string, myStatus Status) {

	conn, err := net.DialTimeout("tcp", addr, *flag_timeout)
	if err != nil {
		//log.Println("      ERR:", err)
		return
	}
	log.Println("                    probing", addr)
	client := rpc.NewClient(conn)

	var stat Status
	err = client.Call("RPC.Status", myStatus, &stat)
	if err == nil {
		pipe <- func() {
			// new peer (not myself): add to peers
			if _, ok := peers[stat.Addr]; !ok {
				log.Println("Found new peer:", stat)
				peers[stat.Addr] = Peer{Client: client}
			} else {
				go client.Close() // don't wait for possibly haning RPC
			}
		}
	}
}
