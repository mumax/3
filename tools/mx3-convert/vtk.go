package main

// Support for vtk 4.2 file output
// Author: Rémy Lassalle-Balier
// Modified by Arne Vansteenkiste, 2012, 2013.
// Modified by Mykola Dvornik, 2013

import (
	"bytes"
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/util"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"io"
	"log"
)

func dumpVTK(out io.Writer, q *data.Slice, meta data.Meta, dataformat string) (err error) {
	err = writeVTKHeader(out, q)
	err = writeVTKCellData(out, q, meta, dataformat)
	err = writeVTKPoints(out, q, dataformat)
	err = writeVTKFooter(out)
	return
}

func writeVTKHeader(out io.Writer, q *data.Slice) (err error) {
	gridsize := q.Mesh().Size()
	_, err = fmt.Fprintln(out, "<?xml version=\"1.0\"?>")
	_, err = fmt.Fprintln(out, "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">")
	_, err = fmt.Fprintf(out, "\t<StructuredGrid WholeExtent=\"0 %d 0 %d 0 %d\">\n", gridsize[Z]-1, gridsize[Y]-1, gridsize[X]-1)
	_, err = fmt.Fprintf(out, "\t\t<Piece Extent=\"0 %d 0 %d 0 %d\">\n", gridsize[Z]-1, gridsize[Y]-1, gridsize[X]-1)
	return
}

func writeVTKPoints(out io.Writer, q *data.Slice, dataformat string) (err error) {
	_, err = fmt.Fprintln(out, "\t\t\t<Points>")
	fmt.Fprintf(out, "\t\t\t\t<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"%s\">\n\t\t\t\t\t", dataformat)
	gridsize := q.Mesh().Size()
	cellsize := q.Mesh().CellSize()
	switch dataformat {
	case "ascii":
		for k := 0; k < gridsize[X]; k++ {
			for j := 0; j < gridsize[Y]; j++ {
				for i := 0; i < gridsize[Z]; i++ {
					x := (float32)(i) * (float32)(cellsize[Z])
					y := (float32)(j) * (float32)(cellsize[Y])
					z := (float32)(k) * (float32)(cellsize[X])
					_, err = fmt.Fprint(out, x, " ", y, " ", z, " ")
				}
			}
		}
	case "binary":
		buffer := new(bytes.Buffer)
		for k := 0; k < gridsize[X]; k++ {
			for j := 0; j < gridsize[Y]; j++ {
				for i := 0; i < gridsize[Z]; i++ {
					x := (float32)(i) * (float32)(cellsize[Z])
					y := (float32)(j) * (float32)(cellsize[Y])
					z := (float32)(k) * (float32)(cellsize[X])
					binary.Write(buffer, binary.LittleEndian, x)
					binary.Write(buffer, binary.LittleEndian, y)
					binary.Write(buffer, binary.LittleEndian, z)
				}
			}
		}
		b64len := uint32(len(buffer.Bytes()))
		bufLen := new(bytes.Buffer)
		binary.Write(bufLen, binary.LittleEndian, b64len)
		base64out := base64.NewEncoder(base64.StdEncoding, out)
		base64out.Write(bufLen.Bytes())
		base64out.Write(buffer.Bytes())
		base64out.Close()
	default:
		log.Fatalf("Illegal VTK data format: %v. Options are: ascii, binary", dataformat)
	}
	_, err = fmt.Fprintln(out, "\n\t\t\t\t</DataArray>")
	_, err = fmt.Fprintln(out, "\t\t\t</Points>")
	return
}

func writeVTKCellData(out io.Writer, q *data.Slice, meta data.Meta, dataformat string) (err error) {
	N := q.NComp()
	data := q.Tensors()
	switch N {
	case 1:
		fmt.Fprintf(out, "\t\t\t<PointData Scalars=\"%s\">\n", meta.Name)
		fmt.Fprintf(out, "\t\t\t\t<DataArray type=\"Float32\" Name=\"%s\" NumberOfComponents=\"%d\" format=\"%s\">\n\t\t\t\t\t", meta.Name, N, dataformat)
	case 3:
		fmt.Fprintf(out, "\t\t\t<PointData Vectors=\"%s\">\n", meta.Name)
		fmt.Fprintf(out, "\t\t\t\t<DataArray type=\"Float32\" Name=\"%s\" NumberOfComponents=\"%d\" format=\"%s\">\n\t\t\t\t\t", meta.Name, N, dataformat)
	case 6, 9:
		fmt.Fprintf(out, "\t\t\t<PointData Tensors=\"%s\">\n", meta.Name)
		fmt.Fprintf(out, "\t\t\t\t<DataArray type=\"Float32\" Name=\"%s\" NumberOfComponents=\"%d\" format=\"%s\">\n\t\t\t\t\t", meta.Name, 9, dataformat) // must be 9!
	default:
		log.Fatalf("vtk: cannot handle %v components", N)
	}
	gridsize := q.Mesh().Size()
	switch dataformat {
	case "ascii":
		for i := 0; i < gridsize[X]; i++ {
			for j := 0; j < gridsize[Y]; j++ {
				for k := 0; k < gridsize[Z]; k++ {
					// if symmetric tensor manage it appart to write the full 9 components
					if N == 6 {
						fmt.Fprint(out, data[util.SwapIndex(0, 9)][i][j][k], " ")
						fmt.Fprint(out, data[util.SwapIndex(1, 9)][i][j][k], " ")
						fmt.Fprint(out, data[util.SwapIndex(2, 9)][i][j][k], " ")
						fmt.Fprint(out, data[util.SwapIndex(1, 9)][i][j][k], " ")
						fmt.Fprint(out, data[util.SwapIndex(3, 9)][i][j][k], " ")
						fmt.Fprint(out, data[util.SwapIndex(4, 9)][i][j][k], " ")
						fmt.Fprint(out, data[util.SwapIndex(2, 9)][i][j][k], " ")
						fmt.Fprint(out, data[util.SwapIndex(4, 9)][i][j][k], " ")
						fmt.Fprint(out, data[util.SwapIndex(5, 9)][i][j][k], " ")
					} else {
						for c := 0; c < N; c++ {
							fmt.Fprint(out, data[util.SwapIndex(c, N)][i][j][k], " ")
						}
					}
				}
			}
		}
	case "binary":
		// Inlined for performance, terabytes of data will pass here...
		buffer := new(bytes.Buffer)
		for i := 0; i < gridsize[X]; i++ {
			for j := 0; j < gridsize[Y]; j++ {
				for k := 0; k < gridsize[Z]; k++ {
					// if symmetric tensor manage it appart to write the full 9 components
					if N == 6 {
						binary.Write(buffer, binary.LittleEndian, data[util.SwapIndex(0, 9)][i][j][k])
						binary.Write(buffer, binary.LittleEndian, data[util.SwapIndex(1, 9)][i][j][k])
						binary.Write(buffer, binary.LittleEndian, data[util.SwapIndex(2, 9)][i][j][k])
						binary.Write(buffer, binary.LittleEndian, data[util.SwapIndex(1, 9)][i][j][k])
						binary.Write(buffer, binary.LittleEndian, data[util.SwapIndex(3, 9)][i][j][k])
						binary.Write(buffer, binary.LittleEndian, data[util.SwapIndex(4, 9)][i][j][k])
						binary.Write(buffer, binary.LittleEndian, data[util.SwapIndex(2, 9)][i][j][k])
						binary.Write(buffer, binary.LittleEndian, data[util.SwapIndex(4, 9)][i][j][k])
						binary.Write(buffer, binary.LittleEndian, data[util.SwapIndex(5, 9)][i][j][k])
					} else {
						for c := 0; c < N; c++ {
							binary.Write(buffer, binary.LittleEndian, data[util.SwapIndex(c, N)][i][j][k])
						}
					}
				}
			}
		}
		b64len := uint32(len(buffer.Bytes()))
		bufLen := new(bytes.Buffer)
		binary.Write(bufLen, binary.LittleEndian, b64len)
		base64out := base64.NewEncoder(base64.StdEncoding, out)
		base64out.Write(bufLen.Bytes())
		base64out.Write(buffer.Bytes())
		base64out.Close()
	default:
		panic(fmt.Errorf("vtk: illegal data format " + dataformat + ". Options are: ascii, binary"))
	}

	fmt.Fprintln(out, "\n\t\t\t\t</DataArray>")
	fmt.Fprintln(out, "\t\t\t</PointData>")
	return
}

func writeVTKFooter(out io.Writer) (err error) {
	_, err = fmt.Fprintln(out, "\t\t</Piece>")
	_, err = fmt.Fprintln(out, "\t</StructuredGrid>")
	_, err = fmt.Fprintln(out, "</VTKFile>")
	return
}
