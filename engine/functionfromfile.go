package engine

import (
	"encoding/csv"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/mumax/3/util"
)

func init() {
	DeclFunc("FunctionFromDatafile", FunctionFromDatafile,
		"Creates an interpolation function using data from two columns in a csv file. "+
			"Arguments: filename, xColumnIdx, yColumnIdx")
}

func isStrictlyIncreasing(x []float64) bool {
	for i := 1; i < len(x); i++ {
		if x[i] <= x[i-1] {
			return false
		}
	}
	return true
}

func LinearInterpolationFunction(xData, yData []float64) func(float64) float64 {

	util.AssertMsg(len(xData) == len(yData), "Interpolation error: given data slices do not have the same length")
	util.AssertMsg(len(xData) != 0, "Interpolation error: data slices are empty")
	util.AssertMsg(isStrictlyIncreasing(xData), "Interpolation error: X values are not strictly increasing")

	return func(x float64) float64 {
		ib := 0 // index for the smallest xData value larger than x
		for ; ib < len(xData); ib++ {
			if x < xData[ib] {
				break
			}
		}

		if ib == 0 {
			return yData[0]
		}

		if ib == len(xData) {
			return yData[len(xData)-1]
		}

		ia := ib - 1 // index for the largest xData value smaller than x
		xa, ya := xData[ia], yData[ia]
		xb, yb := xData[ib], yData[ib]

		return ya + (x-xa)*(yb-ya)/(xb-xa)
	}
}

func FunctionFromDatafile(fname string, xCol, yCol int) func(float64) float64 {
	csvfile, err := os.Open(fname)
	util.FatalErr(err)
	defer csvfile.Close()

	r := csv.NewReader(csvfile)
	r.Comment = '#'

	xData := make([]float64, 0)
	yData := make([]float64, 0)

	for {
		line, err := r.Read()
		if err == io.EOF {
			break
		} else {
			util.FatalErr(err)
		}

		x_, err := strconv.ParseFloat(strings.TrimSpace(line[xCol]), 64)
		util.FatalErr(err)
		y_, err := strconv.ParseFloat(strings.TrimSpace(line[yCol]), 64)
		util.FatalErr(err)

		xData = append(xData, x_)
		yData = append(yData, y_)
	}

	return LinearInterpolationFunction(xData, yData)
}
