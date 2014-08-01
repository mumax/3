package main

import (
	"net/rpc"
)

type Peer struct {
	*rpc.Client
}
