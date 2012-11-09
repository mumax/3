package main

// Support for vtk 4.2 file output
// Author: Rémy Lassalle-Balier
// Modified by Arne Vansteenkiste, aug. 2012.

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/dump"
	"fmt"
	"io"
	"os"
	"unsafe"
)

func dumpVTK(file string, q *dump.Frame, dataformat string) {

	out, err := os.OpenFile(file, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	core.Fatal(err)
	defer out.Close()

	writeVTKHeader(out, q)
	writeVTKPoints(out, q, dataformat)
	writeVTKCellData(out, q, dataformat)
	writeVTKFooter(out)
}

func writeVTKHeader(out io.Writer, q *dump.Frame) {
	gridsize := q.Size()[1:]

	fmt.Fprintln(out, "<?xml version=\"1.0\"?>")
	fmt.Fprintln(out, "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">")
	fmt.Fprintf(out, "\t<StructuredGrid WholeExtent=\"0 %d 0 %d 0 %d\">\n", gridsize[Z]-1, gridsize[Y]-1, gridsize[X]-1)
	fmt.Fprintf(out, "\t\t<Piece Extent=\"0 %d 0 %d 0 %d\">\n", gridsize[Z]-1, gridsize[Y]-1, gridsize[X]-1)
}

func writeVTKPoints(out io.Writer, q *dump.Frame, dataformat string) {
	fmt.Fprintln(out, "\t\t\t<Points>")
	fmt.Fprintf(out, "\t\t\t\t<DataArray type=\"Float32\" Name=\"points\" NumberOfComponents=\"3\" format=\"%s\">\n", dataformat)
	gridsize := q.Size()[1:]
	cellsize := q.MeshStep
	switch dataformat {
	case "ascii":
		for k := 0; k < gridsize[X]; k++ {
			for j := 0; j < gridsize[Y]; j++ {
				for i := 0; i < gridsize[Z]; i++ {
					x := (float32)(i) * (float32)(cellsize[Z])
					y := (float32)(j) * (float32)(cellsize[Y])
					z := (float32)(k) * (float32)(cellsize[X])
					_, err := fmt.Fprint(out, x, " ", y, " ", z, " ")
					core.Fatal(err)
				}
			}
		}
	case "binary":
		// Conversion form float32 [4]byte in big-endian
		// encoding/binary is too slow
		// Inlined for performance, terabytes of data will pass here...
		var bytes []byte
		for k := 0; k < gridsize[X]; k++ {
			for j := 0; j < gridsize[Y]; j++ {
				for i := 0; i < gridsize[Z]; i++ {
					x := (float32)(i) * (float32)(cellsize[Z])
					y := (float32)(j) * (float32)(cellsize[Y])
					z := (float32)(k) * (float32)(cellsize[X])
					bytes = (*[4]byte)(unsafe.Pointer(&x))[:]
					out.Write(bytes)
					bytes = (*[4]byte)(unsafe.Pointer(&y))[:]
					out.Write(bytes)
					bytes = (*[4]byte)(unsafe.Pointer(&z))[:]
					out.Write(bytes)
				}
			}
		}
	default:
		core.Fatal(fmt.Errorf("Illegal VTK data format: %v. Options are: ascii, binary", dataformat))
	}
	fmt.Fprintln(out, "</DataArray>")
	fmt.Fprintln(out, "\t\t\t</Points>")
}

func writeVTKCellData(out io.Writer, q *dump.Frame, dataformat string) {
	N := q.Size()[0]
	data := q.Tensors()
	switch N {
	case 1:
		fmt.Fprintf(out, "\t\t\t<PointData Scalars=\"%s\">\n", q.DataLabel)
		fmt.Fprintf(out, "\t\t\t\t<DataArray type=\"Float32\" Name=\"%s\" NumberOfComponents=\"%d\" format=\"%s\">\n", q.DataLabel, N, dataformat)
	case 3:
		fmt.Fprintf(out, "\t\t\t<PointData Vectors=\"%s\">\n", q.DataLabel)
		fmt.Fprintf(out, "\t\t\t\t<DataArray type=\"Float32\" Name=\"%s\" NumberOfComponents=\"%d\" format=\"%s\">\n", q.DataLabel, N, dataformat)
	case 6, 9:
		fmt.Fprintf(out, "\t\t\t<PointData Tensors=\"%s\">\n", q.DataLabel)
		fmt.Fprintf(out, "\t\t\t\t<DataArray type=\"Float32\" Name=\"%s\" NumberOfComponents=\"%d\" format=\"%s\">\n", q.DataLabel, 9, dataformat) // must be 9!
	default:
		core.Fatal(fmt.Errorf("vtk: cannot handle %v components"))
	}
	gridsize := q.MeshSize
	switch dataformat {
	case "ascii":
		for i := 0; i < gridsize[X]; i++ {
			for j := 0; j < gridsize[Y]; j++ {
				for k := 0; k < gridsize[Z]; k++ {
					// if symmetric tensor manage it appart to write the full 9 components
					if N == 6 {
						fmt.Fprint(out, data[SwapIndex(0, 9)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(1, 9)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(2, 9)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(1, 9)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(3, 9)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(4, 9)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(2, 9)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(4, 9)][i][j][k], " ")
						fmt.Fprint(out, data[SwapIndex(5, 9)][i][j][k], " ")
					} else {
						for c := 0; c < N; c++ {
							fmt.Fprint(out, data[SwapIndex(c, N)][i][j][k], " ")
						}
					}
				}
			}
		}
	case "binary":
		// Inlined for performance, terabytes of data will pass here...
		var bytes []byte
		for i := 0; i < gridsize[X]; i++ {
			for j := 0; j < gridsize[Y]; j++ {
				for k := 0; k < gridsize[Z]; k++ {
					// if symmetric tensor manage it appart to write the full 9 components
					if N == 6 {
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(0, 9)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(1, 9)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(2, 9)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(1, 9)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(3, 9)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(4, 9)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(2, 9)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(4, 9)][i][j][k]))[:]
						out.Write(bytes)
						bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(5, 9)][i][j][k]))[:]
						out.Write(bytes)
					} else {
						for c := 0; c < N; c++ {
							bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(c, N)][i][j][k]))[:]
							out.Write(bytes)
						}
					}
				}
			}
		}
	default:
		panic(fmt.Errorf("vtk: illegal data format " + dataformat + ". Options are: ascii, binary"))
	}
	fmt.Fprintln(out, "</DataArray>")
	fmt.Fprintln(out, "\t\t\t</PointData>")
}

func writeVTKFooter(out io.Writer) {
	fmt.Fprintln(out, "\t\t</Piece>")
	fmt.Fprintln(out, "\t</StructuredGrid>")
	fmt.Fprintln(out, "</VTKFile>")
}
