package main

import (
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
)

// type row struct {
// 	GPUname string
// 	Ncells int32
// 	perf float64
// 	t_nevl float64
// 	str string
// }

var GPUS_DIR = "./gpus/"

func main() {
	gpufiles, _ := os.ReadDir(GPUS_DIR)
	gpus := make(map[string]map[int32]string) // gpus[gpuname][Ncells] = row in .txt
	for i := range gpufiles {
		content, err := os.ReadFile(GPUS_DIR + gpufiles[i].Name())
		check(err)
		var sb strings.Builder
		sb.Write(content)
		text := sb.String()

		gpu := make(map[int32]string) // gpu[Ncells] = row in .txt
		lines := strings.Split(text, "\n")
		for j := range lines {
			fields := strings.Fields(lines[j]) // Split at whitespace
			if len(fields) < 3 {               // Normal rows in benchmark have 3 columns
				continue
			}
			Ncellsfloat, err := strconv.ParseFloat(fields[0], 32)
			check(err)
			Ncells := int32(Ncellsfloat)
			gpu[Ncells] = strings.Join(fields, " ")
		}

		gpuName := strings.TrimSuffix(gpufiles[i].Name(), ".txt")
		gpus[gpuName] = gpu
	}

	// Invert the mapping, to make per-Ncells files
	gpus_inverted := make(map[int32]map[string]string) // gpus_inverted[Ncells][gpuname] = row in .txt
	for gpuname, m := range gpus {
		gpu_inverted := make(map[string]string)
		for Ncells, row := range m {
			gpu_inverted[gpuname] = row
			if gpus_inverted[Ncells] == nil {
				gpus_inverted[Ncells] = make(map[string]string)
			}
			gpus_inverted[Ncells][gpuname] = row + " \"" + gpuname + "\""
		}
	}

	// Create a gpus.txt for every Ncells
	for Ncells, m := range gpus_inverted {
		var rows []string
		for _, row := range m {
			rows = append(rows, row)
		}
		sort.Slice(rows, func(i, j int) bool { // Sort GPUs by ascending performance
			fields_i := strings.Fields(rows[i])         // Split at whitespace
			fields_j := strings.Fields(rows[j])         // Split at whitespace
			if len(fields_i) < 3 || len(fields_j) < 3 { // Normal rows in benchmark have 3 columns
				log.Fatal("Something went horribly wrong, because there are not 3 columns in the generated table.")
			}
			perf_i, erri := strconv.ParseFloat(fields_i[1], 64)
			check(erri)
			perf_j, errj := strconv.ParseFloat(fields_j[1], 64)
			check(errj)
			return perf_i < perf_j
		})
		var sb strings.Builder
		for _, row := range rows {
			sb.WriteString(row)
			sb.WriteString("\n")
		}
		f, err := os.Create("gpus_" + strconv.Itoa(int(Ncells)) + ".txt")
		check(err)
		defer f.Close()
		f.WriteString(sb.String())
	}
	fmt.Println(gpus_inverted)
}

func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
