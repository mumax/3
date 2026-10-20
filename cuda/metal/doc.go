// Package metal provides the Apple Silicon compute runtime used by MuMax3.
//
// The package deliberately exposes a small, CUDA-shaped execution surface:
// shared GPU allocations are represented by their CPU-visible address, kernel
// launches carry CUDA grid and block dimensions, and work is submitted to one
// ordered command queue. Internally, allocation addresses are resolved to an
// MTLBuffer plus byte offset before every launch. This preserves the existing
// data.Slice pointer-and-offset convention while the rest of MuMax3 migrates to
// backend-neutral buffer handles.
//
// Kernel launches, buffer copies, and fills are batched in the current Metal
// command buffer. Sync commits the batch and waits for all earlier work. Host
// reads and writes synchronize automatically because Go memory cannot remain
// borrowed by an asynchronous Metal command.
package metal
