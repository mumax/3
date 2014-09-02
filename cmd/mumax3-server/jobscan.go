package main

import (
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
)

var scan = make(chan struct{})

// RPC-callable method: picks a job of the queue returns it
// for the node to run it.
func (n *Node) ReScan() {
	select {
	default: // already scannning
	case scan <- struct{}{}: // wake-up scanner
	}
}

func RunJobScan(dir string) {
	for {
		<-scan // wait for start sign

		node.lock()
		users := make([]string, 0, len(node.Users))
		for k, _ := range node.Users {
			users = append(users, k)
		}
		node.unlock()

		count := 0
		err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if strings.Contains(path, "/") && strings.HasSuffix(path, ".mx3") && !exist(JobOutputDir(path)) {
				log.Println("found", path)
				node.AddJob(path)
				count++
			}
			return nil
		})
		if err != nil {
			log.Println(err)
		}

		node.lock()
		node.LastJobScanTime = time.Now()
		node.LastJobScanFiles = count
		node.unlock()

	}
}

func exist(filename string) bool {
	_, err := os.Stat(filename)
	return err == nil
}
