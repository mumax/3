package main

import (
	"log"
	"net"
	"os"
	"reflect"
	"runtime"
)

var (
	MyStatus Status
	peers    = make(map[string]Peer)
	jobs     = new(jobList)
	gpus     []GPU
	pipe     = make(chan func(), 1000)
)

type Peer struct {
	Status
}

func MainSlave(laddr string, IPs []string, minPort, maxPort int, jobs []string) {

	// replace laddr by canonical form, as it will serve as unique ID
	h, p, err := net.SplitHostPort(laddr)
	Fatal(err)
	if h == "" {
		h, _ = os.Hostname()
	}
	laddr = net.JoinHostPort(h, p)

	MyStatus = Status{
		Type: "slave",
		Addr: laddr,
	}

	log.Println("Status:", MyStatus)

	go ServeRPC(laddr, RPC{})
	go FindPeers(IPs, minPort, maxPort)
	go ServeHTTP()
	go func() {
		for _, j := range jobs {
			j := j // persistent copy (closure caveat)
			pipe <- func() { AddJob(j) }
		}
	}()
	go func() {
		pipe <- func() {
			gpus = DetectGPUs()
			log.Println("GPUs:", gpus)
		}
	}()

	for {
		log.Println("                    [waiting]")
		f := <-pipe
		log.Println("                    [working]", runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name())
		f()
	}
}

func AddJob(file string) {
	jobs.Push(NewJob(file))
}

// pipe <- f and wait for f() to return
func PipeAndWait(f func()) {
	wait := make(chan struct{})
	pipe <- func() {
		f()
		wait <- struct{}{}
	}
	<-wait
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
