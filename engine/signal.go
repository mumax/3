package engine

import (
    "os"
    "math"
    "strings"
    "bufio"
    "strconv"
)

func init() {
    DeclFunc("LoadSignal", LoadSignal, "...")
    DeclFunc("SignalValue", SignalValue, "...")
    DeclFunc("SetSignalDt", GetSignalDt, "...")
    DeclFunc("GetSignalDt", GetSignalDt, "...")
    DeclFunc("GetSignalLength", GetSignalLength, "...")
    DeclFunc("GetSignalDimensions", GetSignalDimensions, "...")
    DeclFunc("GetSignalDuration", GetSignalDuration, "...")
}

type Signal struct {
    dt      float64
    len     int
    dim     int
    data    []float64
}

func GetSignalDt(S* Signal) float64 {
    return S.dt
}

func GetSignalLength(S* Signal) int {
    return S.len
}

func GetSignalDuration(S* Signal) float64 {
    return float64(S.len) * S.dt
}

func GetSignalDimensions(S* Signal) int {
    return S.dim
}

func SetSignalDt(S* Signal, dt float64) {
    S.dt = dt
}

// Credit to: https://stackoverflow.com/questions/5884154/read-text-file-into-string-array-and-write
func ReadLines(fname string) ([]string, error) {
    file, err := os.Open(fname)

    if err != nil {
        return nil, err
    }

    defer file.Close()

    var lines []string

    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        lines = append(lines, scanner.Text())
    }

    return lines, scanner.Err()
}

func LoadSignal(fname string, dt float64) *Signal {
    lines, err := ReadLines(fname)

    if err != nil {
        panic(err)
    }

    S := new(Signal)

    S.dt   = dt
    S.len  = len(lines)
    S.dim  = len(strings.Fields(lines[0]))
    S.data = make([]float64, S.len * S.dim)

    for j, line := range lines {
        fields := strings.Fields(line)

        for i, field := range fields {
            f, err := strconv.ParseFloat(field, 64)

            if err != nil {
                panic(err)
            }

            S.data[S.dim * j + i] = f
        }
    }

    return S
}


// Signal value at time t, for column n. Uses linear interpolation
// Returns zero if t is longer than the signal duration
func SignalValue(S *Signal, t float64, n int) float64 {
    idxf := math.Floor(t / S.dt)
    idx  := int(idxf)

    if (idx + 1 < S.len && n < S.dim) {

        y0 := S.data[ idx      * S.dim + n]
        y1 := S.data[(idx + 1) * S.dim + n]

        alpha := (t - idxf * S.dt) / S.dt

        return y0 + alpha * (y1 - y0)

    } else {

        return 0.0

    }
}
