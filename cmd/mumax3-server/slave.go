package main

import (
	"log"
	"net"
	"os"
	"reflect"
	"runtime"
)

var (
	pipe     = make(chan func(), 1000)
	peers    = make(map[string]Peer)
	MyStatus Status
)

func MainSlave(laddr string, IPs []string, minPort, maxPort int) {

	// replace laddr by canonical form, as it will serve as unique ID
	h, p, err := net.SplitHostPort(laddr)
	Fatal(err)
	if h == "" {
		h, _ = os.Hostname()
	}
	laddr = net.JoinHostPort(h, p)
	MyStatus = Status{Addr: laddr}
	log.Println("Status:", MyStatus)

	go ServeRPC(laddr, RPC{})
	go FindPeers(IPs, minPort, maxPort)

	for {
		log.Println("                    [waiting]")
		f := <-pipe
		log.Println("                    [working]", runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name())
		f()
	}
}

func (RPC) Status(peerStat Status, ret *Status) error {
	//log.Println("status called by", peerStat)
	*ret = MyStatus

	// Somebody just called my status, let's probe him
	// and possibly add as new peer.
	if peerStat.Addr != MyStatus.Addr {
		pipe <- func() {
			if _, ok := peers[peerStat.Addr]; !ok {
				go func() { RPCProbe(peerStat.Addr) }()
			}
		}
	}

	return nil
}
