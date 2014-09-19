package main

//import "os"
//
//var scan = make(chan struct{})
//
//// RPC-callable method: picks a job of the queue returns it
//// for the node to run it.
//func (n *Node) ReScan() {
//	select {
//	default: // already scannning
//	case scan <- struct{}{}: // wake-up scanner
//	}
//}
//
//func exist(filename string) bool {
//	_, err := os.Stat(filename)
//	return err == nil
//}
