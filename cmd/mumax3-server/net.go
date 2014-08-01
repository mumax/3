package main

import (
	"log"
	"net"
	"net/http"
	"net/rpc"
	"time"
)

const NET_TIMEOUT = 2 * time.Second

// Return a listener in range range host:MinPort, host:MaxPort.
// Fatal if not possible.
//func FindPort(host string) net.Listener {
//	for i := MinPort; i < MaxPort; i++ {
//		if l, err := net.Listen("tcp", fmt.Sprint(host, ":", i)); err == nil {
//			log.Println("listening at", l.Addr())
//			return l
//		}
//	}
//	log.Fatal("no more free ports in range [", MinPort, ",", MaxPort, "[")
//	return nil
//}

func ServeRPC(addr string, service interface{}) {
	rpc.Register(service)
	rpc.HandleHTTP()
	l, err := net.Listen("tcp", addr)
	Fatal(err)
	log.Printf("serving RPC for %T at %v\n", service, l.Addr())
	Fatal(http.Serve(l, nil))
}

// Portscan over all IPs, MinPort-MaxPort combinations.
// When a valid RPC service is found, it is passed to f.
//func RPCScan(f func(string, *rpc.Client)) {
//	for _, ip := range IPs {
//		for port := MinPort; port <= MaxPort; port++ {
//			addr := fmt.Sprint(ip, ":", port)
//			log.Println("RPC scanning", addr)
//			RPCProbe(addr, f)
//		}
//	}
//	log.Println("RPC Scan done")
//}
//
//func RPCProbe(addr string, f func(string, *rpc.Client)) {
//	client, err := Dial(addr)
//	if err != nil {
//		log.Println(err)
//		return
//	}
//	log.Println("RPC Scan found client at", addr)
//	f(addr, client)
//}

// Like rpc.Dial, but with timeout
func Dial(addr string) (*rpc.Client, error) {
	conn, err := net.DialTimeout("tcp", addr, NET_TIMEOUT)
	if err != nil {
		return nil, err
	}
	return rpc.NewClient(conn), nil
}

func Fatal(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
