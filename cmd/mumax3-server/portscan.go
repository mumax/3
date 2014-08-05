package main

import (
	"fmt"
	"log"
	"net"
	"net/rpc"
)

const N_SCANNERS = 128

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
			scanners <- func() { RPCProbe(addr) }
		}
	}
	close(scanners)
	log.Println("Portscan done")
}

//
func RPCProbe(addr string) {
	conn, err := net.DialTimeout("tcp", addr, *flag_timeout)
	if err != nil {
		//log.Println("      ERR:", err)
		return
	}
	//log.Println("                    probing", addr)
	client := rpc.NewClient(conn)
	defer client.Close()

	var peerStat Status
	err = client.Call("RPC.Status", MyStatus, &peerStat)

	if err == nil {
		pipe <- func() {
			// new peer (not myself): add to peers
			if _, ok := peers[peerStat.Addr]; !ok && peerStat.Addr != MyStatus.Addr {
				log.Println("Found new peer:", peerStat)
				peers[peerStat.Addr] = Peer{Status: peerStat}
			}
		}
	}
}
