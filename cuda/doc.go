/*
 Low-level CUDA functionality.
 Inside this package, all data is stored in ZYX coordinates.
 I.e., our X axis (component 0) is perpendicular to the thin film and has usually a small number of cells (or even just one). Z is usually the longest axis. This annoying convention is imposed by the cuda FFT data layout.
*/
package cuda
