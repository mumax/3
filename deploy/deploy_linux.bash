# The cuda versions against which we will compile mumax3
for CUDAVERSION in 7.0 7.5 8.0 9.0 9.1 9.2 10.0 10.1 10.2; do

    # downgrade host compiler for nvcc for old cuda versions
    if [ 1 -eq "$(echo "${CUDAVERSION} < 9.2" | bc)" ]; then
        export NVCC_CCBIN=/usr/bin/gcc-4.8
    else
        export NVCC_CCBIN=/usr/bin/gcc
    fi

    # The final location of the mumax3 executables and libs
    BUILDDIR=./build/mumax3.10_linux_cuda${CUDAVERSION}
    rm -rf $BUILDDIR
    mkdir -p $BUILDDIR
    
    # The location of the home dirctory of this cuda version
    #   We export this variable so that cuda/Makefile knows how to build the wrappers
    export CUDA_HOME=/usr/local/cuda-${CUDAVERSION}
    
    # All supported compute capabilities of this cuda version
    #   We export CUDA_CC so that cuda/Makefile knows what to include in the fat wrappers
    case $CUDAVERSION in
        "7.0")  export CUDA_CC="20 30 32 35 37 50 52 53";;
        "7.5")  export CUDA_CC="20 30 32 35 37 50 52 53";;
        "8.0")  export CUDA_CC="20 30 32 35 37 50 52 53 60 61 62";;
        "9.0")  export CUDA_CC="   30 32 35 37 50 52 53 60 61 62 70";;
        "9.1")  export CUDA_CC="   30 32 35 37 50 52 53 60 61 62 70 72";;
        "9.2")  export CUDA_CC="   30 32 35 37 50 52 53 60 61 62 70 72";;
        "10.0") export CUDA_CC="   30 32 35 37 50 52 53 60 61 62 70 72 75";;
        "10.1") export CUDA_CC="   30 32 35 37 50 52 53 60 61 62 70 72 75";;
        "10.2") export CUDA_CC="   30 32 35 37 50 52 53 60 61 62 70 72 75";;
    esac

    # The path for shared libraries (relative to the build directory)
    RPATH=lib 
    mkdir -p $BUILDDIR/$RPATH
    
    # We overwrite the CGO Flags to make sure that it is compiled against $CUDAVERSION
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export CGO_LDFLAGS="-lcufft -lcurand -lcuda -L${CUDA_HOME}/lib64 -Wl,-rpath -Wl,\$ORIGIN/$RPATH"
    export CGO_CFLAGS="-I${CUDA_HOME}/include -Wl,-rpath -Wl,\$ORIGIN/$RPATH"

    # (Re)build everything
    (cd .. && make realclean && make -j 4 || exit 1)
    
    # Copy the executable and the cuda libraries to the output directory
    cp $GOPATH/bin/mumax3* $BUILDDIR 
    cp $( ldd ${BUILDDIR}/mumax3 | grep libcufft | awk '{print $3}' ) ${BUILDDIR}/${RPATH}
    cp $( ldd ${BUILDDIR}/mumax3 | grep libcurand | awk '{print $3}' ) ${BUILDDIR}/${RPATH}

done
