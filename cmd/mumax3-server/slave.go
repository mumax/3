package main

import (
	"log"
	"net/rpc"
)

// slave runs compute jobs
type slave struct {
	SlaveInfo
	pipe    chan func()
	masters map[string]*masterConn
}

func (s *slave) Main(host string) {
	s.pipe = make(chan func())
	s.masters = make(map[string]*masterConn)
	s.UID = host
	go ServeRPC(host, (*Slave)(s))
	go s.findMasters()

	for {
		log.Println("waiting")
		f := <-s.pipe
		log.Println("working", f)
		f()
	}
}

// slave info sent out to masters
type SlaveInfo struct {
	UID  string
	Addr string
}

// RPC client for a master
type masterConn struct {
	MasterInfo
	*rpc.Client
}

// Public RPC service for slave
type Slave slave

// Called by a master to connect.
// MasterInfo will be stored by slave, SlaveInfo is stored by master.
func (S *Slave) Connect(master MasterInfo, ret *SlaveInfo) error {
	s := (*slave)(S)
	s.addMaster(master)
	*ret = s.SlaveInfo
	return nil
}

func (s *slave) addMaster(master MasterInfo) {
	s.pipe <- func() {
		if _, ok := s.masters[master.UID]; !ok {
			s.masters[master.UID] = &masterConn{MasterInfo: master}
		}
	}
}

func (s *slave) findMasters() {
	log.Println("looking for masters")
	RPCScan(func(addr string, c *rpc.Client) {
		s.pipe <- func() {
			var mInfo MasterInfo
			if err := c.Call("Master.Connect", s.SlaveInfo, &mInfo); err == nil {
				log.Println("Found master @", addr)
				s.masters[mInfo.UID] = &masterConn{MasterInfo: mInfo, Client: c}
			} else {
				log.Println(err)
			}
		}
	})
}

//
//
//func DiscoverServers(baseIP, port string) {
//	log.Println("Scanning for servers")
//	for i := 1; i < 255; i++ {
//		ExploreServer(fmt.Sprint(baseIP, i, port))
//	}
//}
//
//func ExploreServer(addr string) {
//	client, err := rpc.DialHTTP("tcp", addr)
//	if err != nil {
//		return // nothing found
//	}
//
//	var reply struct{}
//	err = client.Call("RPC.Ping", struct{}{}, &reply)
//	if err == nil {
//		pipeline <- func() { AddServer(addr) }
//	} else {
//		log.Println(addr, ":", err)
//	}
//
//}
//
//type RPC struct{}
//
//
//func StartWorkers(n int) {
//	for i := 0; i < n; i++ {
//		go func() {
//			for {
//				(<-workers)()
//			}
//		}()
//	}
//}
