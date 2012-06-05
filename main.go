package main

import (
	"log"
	"nimble-cube/mm"
	"nimble-cube/nc"
	"os"
)

func main() {
	log.SetPrefix("#")
	nc.PrintInfo(os.Stdout)
	mm.Main()
}
