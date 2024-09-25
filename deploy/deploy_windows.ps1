# This script compiles mumax3 for windows 10 against multiple cuda versions.

# The cuda version against which we will compile mumax3
foreach ($CUDA_VERSION in "9.2","10.0","10.1","10.2","11.0") {

    # The final location of executables and libraries ready to be shipped to the user.
    $builddir = "build/mumax3.11_windows_cuda$CUDA_VERSION"

    # The nvidia toolkit installer for cuda 10.2 shoud have set the environment 
    # variable CUDA_PATH_V10_2 which points to the root directory of the 
    # cuda toolbox. (or similar for other cuda versions)
    # This script might not work if this path contains spaces!
    switch ( $CUDA_VERSION ) {
        "9.2"  { $CUDA_HOME = $env:CUDA_PATH_V9_2  }
        "10.0" { $CUDA_HOME = $env:CUDA_PATH_V10_0 }
        "10.1" { $CUDA_HOME = $env:CUDA_PATH_V10_1 }
        "10.2" { $CUDA_HOME = $env:CUDA_PATH_V10_2 }
        "11.0" { $CUDA_HOME = $env:CUDA_PATH_V11_0 }
        default {}
    } 
    if ( -not $CUDA_HOME -or (-not ( Test-Path $CUDA_HOME )) ) {
        Write-Output "CUDA version $CUDA_VERSION does not seem to be installed"
        exit
    }

    # We will compile the kernels for all supported architectures
    switch ( $CUDA_VERSION ) {
        "9.2"  { $CUDA_CC = 30,32,35,37,50,52,53,60,61,62,70,72 }
        "10.0" { $CUDA_CC = 30,32,35,37,50,52,53,60,61,62,70,72,75 }
        "10.1" { $CUDA_CC = 30,32,35,37,50,52,53,60,61,62,70,72,75 }
        "10.2" { $CUDA_CC = 30,32,35,37,50,52,53,60,61,62,70,72,75 }
        "11.0" { $CUDA_CC = 30,32,35,37,50,52,53,60,61,62,70,72,75,80 }
        default {exit}
    } 

    # The NVIDIA compiler which will be used to compile the cuda kernels
    $NVCC = "${CUDA_HOME}/bin/nvcc.exe"
    $CCBIN = "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"
    if ( -not ( Test-Path $CCBIN ) ) {
        Write-Output "CCBIN for nvcc not found at $CCBIN"
        exit
    }

    # overwrite the CGO flags to make sure that mumax3 is compiled against the
    # specified cuda version.
    $env:CGO_LDFLAGS="-lcufft -lcurand -lcuda -L${CUDA_HOME}/lib/x64"
    $env:CGO_CFLAGS="-I${CUDA_HOME}/include -w"

    # Enter the cuda directory to (re)compile the cuda kernels
    Set-Location ../cuda
        Remove-Item *.ptx
        Remove-Item *_wrapper.go
        go build .\cuda2go.go
        $cudafiles = Get-ChildItem -filter "*.cu"
        foreach ($cudafile in $cudafiles) {
            $kernelname = $cudafile.basename
            foreach ($cc in $CUDA_CC) {
                & $NVCC -ccbin ${CCBIN} -Xptxas -O3 -ptx `
                    -gencode="arch=compute_${cc},code=sm_${cc}" `
                    "${cudafile}" -o "${kernelname}_${cc}.ptx"
            }    
            & .\cuda2go $cudafile
            gofmt -w "${kernelname}_wrapper.go"
        }
    Set-Location ../deploy

    # Compile all mumax3 packages and executables
    go install -v "github.com/mumax/3/..."

    # Copy the mumax3 executables and the used cuda libraries to the build directory
    Remove-Item -ErrorAction Ignore -Recurse ${builddir}
    Remove-Item -ErrorAction Ignore "${builddir}.zip"
    New-Item -ItemType "directory" ${builddir} 
    Copy-Item ${env:GOPATH}/bin/mumax3.exe -Destination ${builddir}
    Copy-Item ${env:GOPATH}/bin/mumax3-convert.exe -Destination ${builddir}
    Copy-Item ${env:GOPATH}/bin/mumax3-server.exe -Destination ${builddir}
    Copy-Item ../LICENSE -Destination ${builddir}
    Copy-Item ${CUDA_HOME}/bin/cufft64*.dll -Destination ${builddir}
    Copy-Item ${CUDA_HOME}/bin/curand64*.dll -Destination ${builddir}

    # Finally, put everything in a single archive
    Compress-Archive -Path ${builddir}/* -DestinationPath "${builddir}.zip"
}