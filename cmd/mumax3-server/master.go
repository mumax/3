package main

import (
	"log"
	"net/rpc"
)

// master provides job storage for slaves
type master struct {
	MasterInfo
	pipe   chan func()
	slaves map[string]*slaveConn
}

func (m *master) Main(host string) {
	m.pipe = make(chan func())
	m.slaves = make(map[string]*slaveConn)
	go ServeRPC(host, (*Master)(m))
	go m.findSlaves()

	for {
		log.Println("waiting")
		f := <-m.pipe
		log.Println("working", f)
		f()
	}

}

type MasterInfo struct {
	UID  string
	Addr string
}

type slaveConn struct {
	SlaveInfo
	*rpc.Client
}

// Public PRC service for slave
type Master master

func (M *Master) Connect(slave SlaveInfo, ret *MasterInfo) error {
	m := (*master)(M)
	m.addSlave(slave)
	*ret = m.MasterInfo
	return nil
}

func (m *master) addSlave(slave SlaveInfo) {
	m.pipe <- func() {
		if _, ok := m.slaves[slave.UID]; !ok {
			m.slaves[slave.UID] = &slaveConn{SlaveInfo: slave}
		}
	}
}

func (m *master) findSlaves() {
	log.Println("looking for slaves")
	RPCScan(func(addr string, c *rpc.Client) {
		m.pipe <- func() {
			var sInfo SlaveInfo
			if err := c.Call("Slave.Connect", m.MasterInfo, &sInfo); err == nil {
				log.Println("Found slave @", addr)
				m.slaves[sInfo.UID] = &slaveConn{SlaveInfo: sInfo, Client: c}
			} else {
				log.Println(err)
			}
		}
	})
}
