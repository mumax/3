package main

import (
	"log"
	"time"
)

var runWatchdog = make(chan struct{})

func init(){
	// run watchdog daemon in background
	go func(){
		for{
			<- runWatchdog // wait for start
			DoWatchdog()
		}
	}()
}

func LoopWatchdog() {
	for{
		time.Sleep(3*KeepaliveInterval)
		WakeupWatchdog("")
	}
}

func WakeupWatchdog(string)string{
	select{
		default: return "already running"
		case runWatchdog <- struct{}{}: return "" // ok
	}
}

// single watchdog run:
// re-queues all dead processes
func DoWatchdog() {
		log.Println("Watchdog wake-up")
		WLock()
		defer WUnlock()
		for _, u := range Users {
			for id, j := range u.Jobs {
				if j.IsRunning() && time.Since(j.Alive) > 3*KeepaliveInterval{
					j.Update()
					lastHeartbeat := time.Since(j.Alive)
					if lastHeartbeat > 3*KeepaliveInterval {
						log.Println("job", id, "died, re-queueing")
						j.Reque()
					}
				}
			}
		}
}
