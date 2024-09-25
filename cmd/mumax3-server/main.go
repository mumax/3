package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	_ "net/http/pprof"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
)

var (
	flag_addr     = flag.String("l", ":35360", "Listen and serve at this network address")
	flag_scan     = flag.String("scan", "192.168.0.1-128", "Scan these IP address for other servers")
	flag_ports    = flag.String("ports", "35360-35361", "Scan these ports for other servers")
	flag_timeout  = flag.Duration("timeout", 2*time.Second, "Portscan timeout")
	flag_mumax    = flag.String("exec", "mumax3", "mumax3 executable")
	flag_cachedir = flag.String("cache", "", "mumax3 kernel cache path")
	flag_log      = flag.Bool("log", true, "log debug output")
	flag_halflife = flag.Duration("halflife", 24*time.Hour, "share decay half-life")
)

const (
	MaxIPs            = 1024             // maximum number of IP address to portscan
	N_SCANNERS        = 32               // number of parallel portscan goroutines
	MAXGPU            = 16               // maximum number of GPU's to check for
	KeepaliveInterval = 10 * time.Second // signal process liveness every KeepaliveInterval
)

var (
	thisAddr         string // unique address of this node, e.g., name:1234
	thisHost         string // unique hostname of this node, e.g., name
	IPs              []string
	MinPort, MaxPort int
	global_lock      sync.RWMutex
)

func RLock()   { global_lock.RLock() }
func RUnlock() { global_lock.RUnlock() }
func WLock()   { global_lock.Lock() }
func WUnlock() { global_lock.Unlock() }

const GUI_PORT = 35367 // base port number for GUI (to be incremented by GPU number)

func main() {
	flag.Parse()

	IPs = parseIPs()
	MinPort, MaxPort = parsePorts()

	thisAddr = canonicalAddr(*flag_addr, IPs)
	var err error
	thisHost, _, err = net.SplitHostPort(thisAddr)
	util.FatalErr(err)
	DetectMumax()
	DetectGPUs()
	LoadJobs()

	http.HandleFunc("/do/", HandleRPC)
	http.HandleFunc("/", HandleStatus)
	httpfs.RegisterHandlers()

	// Listen and serve on all interfaces
	go func() {
		log.Println("serving at", thisAddr)

		// Resolve the IPs for thisHost
		thisIP, err := net.LookupHost(thisHost)
		Fatal(err)

		// try to listen and serve on all interfaces other than thisAddr
		// this is for convenience, errors are not fatal.
		_, p, err := net.SplitHostPort(thisAddr)
		Fatal(err)
		ips := util.InterfaceAddrs()
		for _, ip := range ips {
			addr := net.JoinHostPort(ip, p)
			if !contains(thisIP, ip) { // skip thisIP, will start later and is fatal on error
				go func() {
					log.Println("serving at", addr)
					err := http.ListenAndServe(addr, nil)
					if err != nil {
						log.Println("info:", err, "(but still serving other interfaces)")
					}
				}()
			}
		}

		// only on thisAddr, this server's unique address,
		// we HAVE to be listening.
		Fatal(http.ListenAndServe(thisAddr, nil))
	}()

	ProbePeer(thisAddr) // make sure we have ourself as peer
	go FindPeers(IPs, MinPort, MaxPort)
	go RunComputeService()
	go LoopWatchdog()
	go RunShareDecay()

	// re-load jobs every hour so we don't stall on very exceptional circumstances
	go func() {
		for {
			time.Sleep(1 * time.Hour)
			LoadJobs()
		}
	}()

	<-make(chan struct{}) // wait forever
}

// replace laddr by a canonical form, as it will serve as unique ID
func canonicalAddr(laddr string, IPs []string) string {
	// safe initial guess: hostname:port
	h, p, err := net.SplitHostPort(laddr)
	Fatal(err)
	if h == "" {
		h, _ = os.Hostname()
	}
	name := net.JoinHostPort(h, p)

	ips := util.InterfaceAddrs()
	for _, ip := range ips {
		if contains(IPs, ip) {
			return net.JoinHostPort(ip, p)

		}
	}

	return name
}

func contains(arr []string, x string) bool {
	for _, s := range arr {
		if x == s {
			return true
		}
	}
	return false
}

// Parse port range flag. E.g.:
//
//	1234-1237 -> 1234, 1237
func parsePorts() (minPort, maxPort int) {
	p := *flag_ports
	split := strings.Split(p, "-")
	if len(split) > 2 {
		log.Fatal("invalid port range:", p)
	}
	minPort, _ = strconv.Atoi(split[0])
	if len(split) > 1 {
		maxPort, _ = strconv.Atoi(split[1])
	}
	if maxPort == 0 {
		maxPort = minPort
	}
	if minPort == 0 || maxPort == 0 || maxPort < minPort {
		log.Fatal("invalid port range:", p)
	}
	return
}

// init IPs from flag
func parseIPs() []string {
	var IPs []string
	defer func() {
		if err := recover(); err != nil {
			log.Fatal("invalid IP range:", *flag_scan)
		}
	}()

	p := *flag_scan
	split := strings.Split(p, ",")
	for _, s := range split {
		split := strings.Split(s, ".")
		if len(split) != 4 {
			log.Fatal("invalid IP address range:", s)
		}
		var start, stop [4]uint
		for i, s := range split {
			split := strings.Split(s, "-")
			first := atobyte(split[0])
			start[i], stop[i] = first, first
			if len(split) > 1 {
				stop[i] = atobyte(split[1])
			}
		}

		for A := start[0]; A <= stop[0]; A++ {
			for B := start[1]; B <= stop[1]; B++ {
				for C := start[2]; C <= stop[2]; C++ {
					for D := start[3]; D <= stop[3]; D++ {
						if len(IPs) > MaxIPs {
							log.Fatal("too many IP addresses to scan in", p)
						}
						IPs = append(IPs, fmt.Sprintf("%v.%v.%v.%v", A, B, C, D))
					}
				}
			}
		}
	}
	return IPs
}

func atobyte(a string) uint {
	i, err := strconv.Atoi(a)
	if err != nil {
		panic(err)
	}
	if int(byte(i)) != i {
		panic("too large")
	}
	return uint(i)
}
