package main

import (
	"log"
	"net"
	"net/rpc"
)

type RPC struct{}

func ServeRPC(addr string, service interface{}) {
	server := rpc.NewServer()
	server.Register(service)
	l, err := net.Listen("tcp", addr)
	Fatal(err)
	log.Printf("serving RPC for %T at %v\n", service, l.Addr())
	server.Accept(l)
}
