package main

import (
	"log"
	"net"
	"net/http"
	"net/rpc"
)

type RPC struct{}

func ServeRPC(addr string, service interface{}) {
	rpc.Register(service)
	rpc.HandleHTTP()
	l, err := net.Listen("tcp", addr)
	Fatal(err)
	log.Printf("serving RPC for %T at %v\n", service, l.Addr())
	Fatal(http.Serve(l, nil))
}
