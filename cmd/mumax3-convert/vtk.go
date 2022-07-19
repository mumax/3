package main

// Support for vtk 4.2 file output
// Author: Rémy Lassalle-Balier
// Modified by Arne Vansteenkiste, 2012, 2013.
// Modified by Mykola Dvornik, 2013

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"io"
	"log"

	"github.com/mumax/3/v3/data"
)

func dumpVTK(out io.Writer, q *data.Slice, meta data.Meta, dataformat string) (err error) {
	err = writeVTKHeader(out, q)
	err = writeVTKCellData(out, q, meta, dataformat)
	err = writeVTKPoints(out, q, dataformat, meta)
	err = writeVTKFooter(out)
	return
}

func writeVTKHeader(out io.Writer, q *data.Slice) (err error) {
	gridsize := q.Size()
	_, err = fmt.Fprintln(out, "<?xml version=\"1.0\"?>")
	_, err = fmt.Fprintln(out, "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">")
	_, err = fmt.Fprintf(out, "\t<StructuredGrid WholeExtent=\"0 %d 0 %d 0 %d\">\n", gridsize[0]-1, gridsize[1]-1, gridsize[2]-1)
	_, err = fmt.Fprintf(out, "\t\t<Piece Extent=\"0 %d 0 %d 0 %d\">\n", gridsize[0]-1, gridsize[1]-1, gridsize[2]-1)
	return
}

func writeVTKPoints(out io.Writer, q *data.Slice, dataformat string, info data.Meta) (err error) {
	_, err = fmt.Fprintln(out, "\t\t\t<Points>")
	fmt.Fprintf(out, "\t\t\t\t<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"%s\">\n\t\t\t\t\t", dataformat)
	gridsize := q.Size()
	cellsize := info.CellSize
	switch dataformat {
	case "ascii":
		for k := 0; k < gridsize[2]; k++ {
			for j := 0; j < gridsize[1]; j++ {
				for i := 0; i < gridsize[0]; i++ {
					x := (float32)(i) * (float32)(cellsize[0])
					y := (float32)(j) * (float32)(cellsize[1])
					z := (float32)(k) * (float32)(cellsize[2])
					_, err = fmt.Fprint(out, x, " ", y, " ", z, " ")
				}
			}
		}
	case "binary":
		buffer := new(bytes.Buffer)
		for k := 0; k < gridsize[2]; k++ {
			for j := 0; j < gridsize[1]; j++ {
				for i := 0; i < gridsize[0]; i++ {
					x := (float32)(i) * (float32)(cellsize[0])
					y := (float32)(j) * (float32)(cellsize[1])
					z := (float32)(k) * (float32)(cellsize[2])
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
	gridsize := q.Size()
	switch dataformat {
	case "ascii":
		for k := 0; k < gridsize[2]; k++ {
			for j := 0; j < gridsize[1]; j++ {
				for i := 0; i < gridsize[0]; i++ {
					// if symmetric tensor manage it appart to write the full 9 components
					if N == 6 {
						fmt.Fprint(out, data[0][k][j][i], " ")
						fmt.Fprint(out, data[1][k][j][i], " ")
						fmt.Fprint(out, data[2][k][j][i], " ")
						fmt.Fprint(out, data[1][k][j][i], " ")
						fmt.Fprint(out, data[3][k][j][i], " ")
						fmt.Fprint(out, data[4][k][j][i], " ")
						fmt.Fprint(out, data[2][k][j][i], " ")
						fmt.Fprint(out, data[4][k][j][i], " ")
						fmt.Fprint(out, data[5][k][j][i], " ")
					} else {
						for c := 0; c < N; c++ {
							fmt.Fprint(out, data[c][k][j][i], " ")
						}
					}
				}
			}
		}
	case "binary":
		// Inlined for performance, terabytes of data will pass here...
		buffer := new(bytes.Buffer)
		for k := 0; k < gridsize[2]; k++ {
			for j := 0; j < gridsize[1]; j++ {
				for i := 0; i < gridsize[0]; i++ {
					// if symmetric tensor manage it appart to write the full 9 components
					if N == 6 {
						binary.Write(buffer, binary.LittleEndian, data[0][k][j][i])
						binary.Write(buffer, binary.LittleEndian, data[1][k][j][i])
						binary.Write(buffer, binary.LittleEndian, data[2][k][j][i])
						binary.Write(buffer, binary.LittleEndian, data[1][k][j][i])
						binary.Write(buffer, binary.LittleEndian, data[3][k][j][i])
						binary.Write(buffer, binary.LittleEndian, data[4][k][j][i])
						binary.Write(buffer, binary.LittleEndian, data[2][k][j][i])
						binary.Write(buffer, binary.LittleEndian, data[4][k][j][i])
						binary.Write(buffer, binary.LittleEndian, data[5][k][j][i])
					} else {
						for c := 0; c < N; c++ {
							binary.Write(buffer, binary.LittleEndian, data[c][k][j][i])
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
