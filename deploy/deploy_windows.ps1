# This script compiles mumax3 for windows 10 against multiple cuda versions.

param ( # Optional arguments. Example usage: ./deploy_windows.ps1 -CUDA_VERSIONS 12.6 -CUDA_CC 86
    [String[]]$CUDA_VERSIONS = ("10.0","10.1","10.2","11.0","11.1","11.8","12.0","12.6"), # The cuda versions against which we will compile mumax3
    [Int[]]$CUDA_CC
)

foreach ($CUDA_VERSION_STR in $CUDA_VERSIONS ) {
    # The final location of executables and libraries ready to be shipped to the user.
    $builddir = "build/mumax3.11_windows_cuda$CUDA_VERSION_STR"

    # The nvidia toolkit installer for CUDA 12.6 should have set the environment 
    # variable CUDA_PATH_V12_6 which points to the root directory of the 
    # CUDA toolbox (or similar for other CUDA versions).
    $CUDA_VERSION = [Version]::Parse($CUDA_VERSION_STR) # Convert to Version type for easy handling and comparison
    $CUDA_ENV_VAR_NAME = "CUDA_PATH_V$($CUDA_VERSION.Major)_$($CUDA_VERSION.Minor)"
    $CUDA_HOME = [Environment]::GetEnvironmentVariable($CUDA_ENV_VAR_NAME, "Machine") # "Machine" for system-wide environment variables
    if ( -not $CUDA_HOME -or (-not ( Test-Path $CUDA_HOME )) ) {
        Write-Output "CUDA version $CUDA_VERSION_STR does not seem to be installed"
        Write-Output "(system-wide environment variable $CUDA_ENV_VAR_NAME does not exist or points to nonexistent directory)"
        exit
    }

    #! SUBSTITUTE YOUR OWN PATH TO cl.exe BELOW
    # Not every CUDA version is compatible with any Visual C/C++ version: compiling for CUDA <11.6 requires VS <=2017.
    # See VS/CUDA compatibility matrix at https://quasar.ugent.be/files/doc/cuda-msvc-compatibility.html (with old VS downloads available).
    $VS2022 = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64" # Supported by CUDA v11.6-v12.*
    $VS2017 = "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64" # Supported by CUDA v8.0-v12.*
    switch ( $CUDA_VERSION ) {
        {$_ -lt [Version]::new(11.6)} { $CCBIN = $VS2017 }
        {$_ -ge [Version]::new(11.6)} { $CCBIN = if ($VS2022) {$VS2022} else {$VS2017} } # Use VS2017 if 2022 not installed
        default { Write-Output "Failed to parse CUDA version $CUDA_VERSION_STR" }
    }
    if ( -not ( Test-Path $CCBIN ) ) {
        Write-Output "CCBIN for nvcc not found at $CCBIN"
        exit
    }

    # We will compile the kernels for all supported architectures
    # See supported CC for each CUDA version at https://stackoverflow.com/a/28933055
    # See min. driver version for each CUDA version at https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility
    if ( -not $CUDA_CC ) {
        switch ( $CUDA_VERSION_STR ) {
            "10.0" { $CUDA_CC = 50,52,53,60,61,62,70,72,75 } # Min. Windows driver: >=411.31
            "10.1" { $CUDA_CC = 50,52,53,60,61,62,70,72,75 } # Min. Windows driver: >=418.96
            "10.2" { $CUDA_CC = 50,52,53,60,61,62,70,72,75 } # Min. Windows driver: >=441.22
            "11.0" { $CUDA_CC = 50,52,53,60,61,62,70,72,75,80 } # Min. Windows driver: >=452.39
            "11.1" { $CUDA_CC = 50,52,53,60,61,62,70,72,75,80,86 } # Min. Windows driver: >=452.39 (Same CC for 11.1-11.7)
            "11.8" { $CUDA_CC = 50,52,53,60,61,62,70,72,75,80,86,87,89 } # Min. Windows driver: >=452.39
            "12.0" { $CUDA_CC = 50,52,53,60,61,62,70,72,75,80,86,87,89,90 } # Min. Windows driver: >=527.41 (Same CC for all 12.x.)
            "12.6" { $CUDA_CC = 50,52,53,60,61,62,70,72,75,80,86,87,89,90 } # Min. Windows driver: >=527.41 (Same CC for all 12.x.)
            default {exit}
        }
    }

    # The NVIDIA compiler which will be used to compile the cuda kernels
    $NVCC = "${CUDA_HOME}/bin/nvcc.exe"
    
    # overwrite the CGO flags to make sure that mumax3 is compiled against the
    # specified cuda version.
    $env:CGO_LDFLAGS="-lcufft -lcurand -lcuda -L `"$CUDA_HOME/lib/x64`""
    $env:CGO_CFLAGS="-I `"$CUDA_HOME/include`" -w"

    # Enter the cuda directory to (re)compile the cuda kernels
    Set-Location ../cuda
        go build .\cuda2go.go
        $cudafiles = Get-ChildItem -filter "*.cu"
        foreach ($cudafile in $cudafiles) {
            $kernelname = $cudafile.basename
            Remove-Item "${kernelname}_*.ptx"
            Remove-Item "${kernelname}_*wrapper.go"
            foreach ($cc in $CUDA_CC) {
                & $NVCC -ccbin "`"${CCBIN}`"" -Xptxas -O3 -ptx `
                    -gencode="arch=compute_${cc},code=sm_${cc}" `
                    "${cudafile}" -o "${kernelname}_${cc}.ptx"
            }
            & .\cuda2go $cudafile
            gofmt -w "${kernelname}_wrapper.go"
        }
    Set-Location ../deploy

    # Compile all mumax3 packages and executables. Determine the commit hash and pass it along.
    $COMMIT_HASH = git rev-parse --short HEAD 2>$null
    if (-not $COMMIT_HASH) {
        $COMMIT_HASH = "unknown"
        Write-Host "Warning: Could not determine Git commit hash. Using 'unknown'."
    }
    go install -ldflags "-X main.commitHash=$COMMIT_HASH" -v "github.com/mumax/3/..."

    # Copy the mumax3 executables and the used cuda libraries to the build directory
    Remove-Item -ErrorAction Ignore -Recurse ${builddir}
    Remove-Item -ErrorAction Ignore "${builddir}.zip"
    New-Item -ItemType "directory" ${builddir} 
    Copy-Item ${env:GOPATH}/bin/mumax3.exe -Destination ${builddir}
    Copy-Item ${env:GOPATH}/bin/mumax3-convert.exe -Destination ${builddir}
    Copy-Item ${env:GOPATH}/bin/mumax3-server.exe -Destination ${builddir}
    Copy-Item ${env:GOPATH}/bin/mumax3-format.exe -Destination ${builddir}
    Copy-Item ../LICENSE -Destination ${builddir}
    Copy-Item ${CUDA_HOME}/bin/cufft64*.dll -Destination ${builddir}
    Copy-Item ${CUDA_HOME}/bin/curand64*.dll -Destination ${builddir}

    # Finally, put everything in a single archive
    Compress-Archive -Path ${builddir}/* -DestinationPath "${builddir}.zip"
}