package main

import (
	"cmp"
	"fmt"
	"runtime"
	"strings"
	"syscall"
	"unsafe"
)

// Read the registry to retrieve relevant CPU info.
// This relies on the syscall package, which was frozen in Go 1.3. Nonetheless,
// the method used here still works on Windows on Go 1.26, and the alternatives
// are far less preferable for a feature as minor as retrieving some CPU info:
//   - Calling the PowerShell replacement for wmic: starting a PowerShell takes several seconds
//   - Supplemental or third-party Go packages: mumax³ currently has zero external dependencies
func getWindowsCPUInfo() string {
	var info strings.Builder
	info.WriteString("CPU info: ")

	// CPU name
	name, err := readRegString(`HARDWARE\DESCRIPTION\System\CentralProcessor\0`, "ProcessorNameString")
	if err != nil {
		info.WriteString("Unknown")
	} else {
		info.WriteString(name)
	}

	// Number of logical cores
	fmt.Fprintf(&info, ", Cores: %d", runtime.NumCPU())

	// Standard clock speed
	MHz, err := readRegDWord(`HARDWARE\DESCRIPTION\System\CentralProcessor\0`, "~MHz")
	if err == nil {
		fmt.Fprintf(&info, ", %d MHz", MHz)
	}

	return info.String()
}

// Read a SZ-type value from the registry.
func readRegString(key string, value string) (string, error) {
	k, errk := syscall.UTF16PtrFromString(key)
	v, errv := syscall.UTF16PtrFromString(value)
	err := cmp.Or(errk, errv)
	if err != nil {
		return "", err
	}

	var h syscall.Handle
	err = syscall.RegOpenKeyEx(syscall.HKEY_LOCAL_MACHINE, k, 0, syscall.KEY_READ, &h)
	defer syscall.RegCloseKey(h)
	if err != nil {
		return "", err
	}

	var typ, n uint32
	err = syscall.RegQueryValueEx(h, v, nil, &typ, nil, &n)
	if err != nil {
		return "", err
	}

	buf := make([]uint16, n/2)
	err = syscall.RegQueryValueEx(h, v, nil, &typ, (*byte)(unsafe.Pointer(&buf[0])), &n)
	if err != nil {
		return "", err
	}

	if typ != syscall.REG_SZ {
		return "", fmt.Errorf("unexpected registry type %d", typ)
	}

	return syscall.UTF16ToString(buf), nil
}

// Read a DWORD-type value from the registry.
func readRegDWord(key string, value string) (uint32, error) {
	k, errk := syscall.UTF16PtrFromString(key)
	v, errv := syscall.UTF16PtrFromString(value)
	err := cmp.Or(errk, errv)
	if err != nil {
		return 0, err
	}

	var h syscall.Handle
	err = syscall.RegOpenKeyEx(syscall.HKEY_LOCAL_MACHINE, k, 0, syscall.KEY_READ, &h)
	defer syscall.RegCloseKey(h)
	if err != nil {
		return 0, err
	}

	var typ, data uint32
	var n uint32 = 4
	err = syscall.RegQueryValueEx(h, v, nil, &typ, (*byte)(unsafe.Pointer(&data)), &n)
	if err != nil {
		return 0, err
	}

	if typ != syscall.REG_DWORD {
		return 0, fmt.Errorf("unexpected registry type %d", typ)
	}

	return data, nil
}
