package cu

// This file implements CUDA unified addressing.

//#include <cuda.h>
import "C"

import (
	"unsafe"
)

// Make allocations from the peer Context available to the current context.
func CtxEnablePeerAccess(peer Context) {
	err := Result(C.cuCtxEnablePeerAccess(C.CUcontext(unsafe.Pointer(uintptr(peer))), C.uint(0)))
	if err != SUCCESS {
		panic(err)
	}
}

// Make allocations from the peer Context available to the current context.
func (peer Context) EnablePeerAccess() {
	CtxEnablePeerAccess(peer)
}

// Reverses CtxEnablePeerAccess().
func CtxDisablePeerAccess(peer Context) {
	err := Result(C.cuCtxDisablePeerAccess(C.CUcontext(unsafe.Pointer(uintptr(peer)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Reverses EnablePeerAccess().
func (peer Context) DisablePeerAccess() {
	CtxDisablePeerAccess(peer)
}

// Returns true if CtxEnablePeerAccess can be called on a context for dev and peerDev.
func DeviceCanAccessPeer(dev, peer Device) bool {
	var canAccessPeer C.int
	err := Result(C.cuDeviceCanAccessPeer(&canAccessPeer, C.CUdevice(dev), C.CUdevice(peer)))
	if err != SUCCESS {
		panic(err)
	}
	return int(canAccessPeer) != 0
}

// Returns true if CtxEnablePeerAccess can be called on a context for dev and peerDev.
func (dev Device) CanAccessPeer(peer Device) bool {
	return DeviceCanAccessPeer(dev, peer)
}
