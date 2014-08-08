package main

import (
	"log"
	"net"
	"net/rpc"
)

type RPC struct{ node *Node }

// go serve rpc for service in a separate goroutine.
// returns when the rpc is up and running.
func GoRunRPCService(addr string, service interface{}) {
	server := rpc.NewServer()
	server.Register(service)
	l, err := net.Listen("tcp", addr)
	Fatal(err)
	log.Printf("serving RPC for %T at %v\n", service, l.Addr())
	go server.Accept(l)
}

func (r *RPC) Ping(peerInf NodeInfo, myInf *NodeInfo) error {
	n := r.node
	*myInf = n.Info()

	// Somebody just called my status, let's probe him
	// and possibly add as new peer.
	if _, ok := n.PeerInfo(peerInf.Addr); !ok {
		n.AddPeer(peerInf)
		//ProbePeer(peerInf.Addr)
	}
	return nil
}
