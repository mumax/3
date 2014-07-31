package main

import (
	"flag"
	"fmt"
	"log"
	"strconv"
	"strings"
)

var (
	flag_slave = flag.Bool("slave", false, "Run as compute node")
	flag_ip    = flag.String("ip", "", "Serve at this IP address, blank means auto")
	flag_scan  = flag.String("scan", "192.168.0-1.1-64,127.0.0.1", "Scan these IP address for other servers")
	flag_ports = flag.String("ports", "35360-35369", "Scan these ports for other servers")
)

var (
	MinPort, MaxPort int      // port range to use and portscan
	IPs              []string // all IP addresses to portscan
)

const MaxIPs = 1024

func main() {
	log.SetFlags(0)
	log.SetPrefix("mumax3-server: ")
	flag.Parse()

	parsePorts()
	parseIPs()

	if *flag_slave {
		new(slave).Main(*flag_ip)
	} else {
		new(master).Main(*flag_ip)
	}
}

// init MinPort, MaxPort from CLI flag
func parsePorts() {
	p := *flag_ports
	split := strings.Split(p, "-")
	if len(split) != 2 {
		log.Fatal("invalid port range:", p)
	}
	MinPort, _ = strconv.Atoi(split[0])
	MaxPort, _ = strconv.Atoi(split[1])
	if MinPort == 0 || MaxPort == 0 || MaxPort < MinPort {
		log.Fatal("invalid port range:", p)
	}
}

// init IPs from flag
func parseIPs() {
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
		var start, stop [4]byte
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
}

func atobyte(a string) byte {
	i, err := strconv.Atoi(a)
	if err != nil {
		panic(err)
	}
	if int(byte(i)) != i {
		panic("too large")
	}
	return byte(i)
}
