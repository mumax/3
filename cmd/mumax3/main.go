// mumax3 main command
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path"
	"runtime"
	"strings"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/engine"
	"github.com/mumax/3/script"
	"github.com/mumax/3/util"
)

var (
	flag_failfast = flag.Bool("failfast", false, "If one simulation fails, stop entire batch immediately")
	flag_test     = flag.Bool("test", false, "Cuda test (internal)")
	flag_version  = flag.Bool("v", true, "Print version")
	flag_vet      = flag.Bool("vet", false, "Check input files for errors, but don't run them")
	// more flags in engine/gofiles.go
	commitHash string
)

func main() {
	flag.Parse()
	log.SetPrefix("")
	log.SetFlags(0)

	cuda.Init(*engine.Flag_gpu)

	cuda.Synchronous = *engine.Flag_sync
	if *flag_version {
		printVersion()
	}

	// used by bootstrap launcher to test cuda
	// successful exit means cuda was initialized fine
	if *flag_test {
		os.Exit(0)
	}

	defer engine.Close() // flushes pending output, if any

	if *flag_vet {
		vet()
		return
	}

	switch flag.NArg() {
	case 0:
		if *engine.Flag_interactive {
			runInteractive()
		}
	case 1:
		runFileAndServe(flag.Arg(0))
	default:
		RunQueue(flag.Args())
	}
}

func runInteractive() {
	fmt.Println("//no input files: starting interactive session")
	//initEngine()

	// setup outut dir
	now := time.Now()
	outdir := fmt.Sprintf("mumax-%v-%02d-%02d_%02dh%02d.out", now.Year(), int(now.Month()), now.Day(), now.Hour(), now.Minute())
	engine.InitIO(outdir, outdir, *engine.Flag_forceclean)

	engine.Timeout = 365 * 24 * time.Hour // basically forever

	// set up some sensible start configuration
	engine.Eval(`SetGridSize(128, 64, 1)
		SetCellSize(4e-9, 4e-9, 4e-9)
		Msat = 1e6
		Aex = 10e-12
		alpha = 1
		m = RandomMag()`)
	addr := goServeGUI()
	openbrowser("http://127.0.0.1" + addr)
	engine.RunInteractive()
}

func runFileAndServe(fname string) {
	if path.Ext(fname) == ".go" {
		runGoFile(fname)
	} else {
		runScript(fname)
	}
}

func runScript(fname string) {
	outDir := util.NoExt(fname) + ".out"
	if *engine.Flag_od != "" {
		outDir = *engine.Flag_od
	}
	engine.InitIO(fname, outDir, *engine.Flag_forceclean)

	fname = engine.InputFile

	var code *script.BlockStmt
	var err2 error
	if fname != "" {
		// first we compile the entire file into an executable tree
		code, err2 = engine.CompileFile(fname)
		util.FatalErr(err2)
	}

	// now the parser is not used anymore so it can handle web requests
	goServeGUI()

	if *engine.Flag_interactive {
		openbrowser("http://127.0.0.1" + *engine.Flag_port)
	}

	// start executing the tree, possibly injecting commands from web gui
	engine.EvalFile(code)

	if *engine.Flag_interactive {
		engine.RunInteractive()
	}
}

func runGoFile(fname string) {

	// pass through flags
	flags := []string{"run", fname}
	flag.Visit(func(f *flag.Flag) {
		if f.Name != "o" {
			flags = append(flags, fmt.Sprintf("-%v=%v", f.Name, f.Value))
		}
	})

	if *engine.Flag_od != "" {
		flags = append(flags, fmt.Sprintf("-o=%v", *engine.Flag_od))
	}

	cmd := exec.Command("go", flags...)
	log.Println("go", flags)
	cmd.Stdout = os.Stdout
	cmd.Stdin = os.Stdin
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		os.Exit(1)
	}
}

// start Gui server and return server address
func goServeGUI() string {
	if *engine.Flag_port == "" {
		log.Println(`//not starting GUI (-http="")`)
		return ""
	}
	addr := engine.GoServe(*engine.Flag_port)
	fmt.Print("//starting GUI at http://127.0.0.1", addr, "\n")
	return addr
}

// print version to stdout
func printVersion() {
	engine.LogOut(engine.UNAME)
	engine.LogOut(fmt.Sprintf("commit hash: %s", commitHash))
	engine.LogOut(getCPUInfo())
	engine.LogOut(fmt.Sprintf("GPU info: %s, using cc=%d PTX", cuda.GPUInfo, cuda.UseCC))
	osInfo := fmt.Sprintf("OS  info: %s, Hostname: %s", getOSInfo(), getHostname())
	engine.LogOut(osInfo)
	engine.LogOut(fmt.Sprintf("Timestamp: %s", time.Now().Format("2006-01-02 15:04:05")))
	engine.LogOut("(c) Arne Vansteenkiste, Dynamat LAB, Ghent University, Belgium")
	engine.LogOut("This is free software without any warranty. See license.txt")
	engine.LogOut("********************************************************************//")
	engine.LogOut("  If you use mumax in any work or publication,                      //")
	engine.LogOut("  we kindly ask you to cite the references in references.bib        //")
	engine.LogOut("********************************************************************//")
}

func getHostname() string {
	hostname, err := os.Hostname()
	if err != nil {
		return "Unknown"
	}
	return hostname
}

func getOSInfo() string {
	// Check the runtime operating system
	switch runtime.GOOS {
	case "windows":
		return "Windows OS"
	case "linux":
		return getLinuxOSInfo()
	// Add more cases for other operating systems if needed
	default:
		return fmt.Sprintf("Unknown OS: %s", runtime.GOOS)
	}
}

func getLinuxOSInfo() string {
	// Check if the file exists
	file, err := os.Open("/etc/os-release")
	if err != nil {
		return fmt.Sprintf("Unknown OS, Error: %s", err.Error())
	}
	defer file.Close()

	// Scan the file line by line
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 && parts[0] == "PRETTY_NAME" {
			// Remove surrounding quotes and return
			return strings.Trim(parts[1], `"`)
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Sprintf("Unknown OS, Error: %s", err.Error())
	}

	return "Unknown OS"
}

func getCPUInfo() string {
	// Check the runtime operating system
	switch runtime.GOOS {
	case "windows":
		return getWindowsCPUInfo()
	case "linux":
		return getLinuxCPUInfo()
	// Add more cases for other operating systems if needed
	default:
		return fmt.Sprintf("CPU info: Unknown OS: %s", runtime.GOOS)
	}
}

func getWindowsCPUInfo() string {
	// Get CPU model name
	cmd := exec.Command("wmic", "cpu", "get", "Name")
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return fmt.Sprintf("CPU info: Unknown, Error: %s", err.Error())
	}
	output := strings.Split(out.String(), "\n")
	cpuModel := "Unknown model"
	if len(output) > 1 {
		cpuModel = strings.TrimSpace(output[1])
	}

	// Get CPU number of cores
	cpuCores := runtime.NumCPU()

	// Get CPU speed
	cmd = exec.Command("wmic", "cpu", "get", "MaxClockSpeed")
	out.Reset()
	cmd.Stdout = &out
	cpuMHz := "Unknown clock frequency"
	if err := cmd.Run(); err == nil {
		output = strings.Split(out.String(), "\n")
		if len(output) > 1 {
			cpuMHz = strings.TrimSpace(output[1]) + " MHz"
		}
	}

	return fmt.Sprintf("CPU info: %s, Cores: %d, %s", cpuModel, cpuCores, cpuMHz)
}

func getLinuxCPUInfo() string {
	file, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return fmt.Sprintf("CPU info: Unknown, Error: %s", err.Error())
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var cpuDetails []string
	var cpuModel, cpuCores, cpuMHz string
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Split(line, ":")
		if len(fields) != 2 {
			continue
		}
		key := strings.TrimSpace(fields[0])
		value := strings.TrimSpace(fields[1])
		switch key {
		case "model name":
			cpuModel = value
		case "cpu cores":
			cpuCores = value
		case "cpu MHz":
			cpuMHz = value
		}
	}
	if cpuModel != "" && cpuCores != "" && cpuMHz != "" {
		cpuDetails = append(cpuDetails, fmt.Sprintf("CPU info: %s, Cores: %s, MHz: %s", cpuModel, cpuCores, cpuMHz))
	}

	return strings.Join(cpuDetails, "; ")
}
