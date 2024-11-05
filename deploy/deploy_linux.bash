# The cuda versions against which we will compile mumax3
for CUDAVERSION in 10.0 10.1 10.2 11.0 11.1 11.8 12.0 12.6; do

    #! NOTE: each CUDA version has a MAXIMUM GCC version: https://stackoverflow.com/a/46380601
    #! EDIT IF-ELSE BELOW TO REFER TO YOUR INSTALLED GCC VERSION(S)!
    if [ 1 -eq "$(echo "${CUDAVERSION} < 9.2" | bc)" ]; then
        export NVCC_CCBIN=/usr/bin/gcc-4.8
    else
        export NVCC_CCBIN=/usr/bin/gcc
    fi

    # The final location of the mumax3 executables and libs
    MUMAX3UNAME=mumax3.11_linux_cuda${CUDAVERSION}
    BUILDDIR=./build/${MUMAX3UNAME}
    rm -rf $BUILDDIR
    mkdir -p $BUILDDIR
    
    # The location of the home dirctory of this cuda version
    #   We export this variable so that cuda/Makefile knows how to build the wrappers
    export CUDA_HOME=/usr/local/cuda-${CUDAVERSION}
    
    # All supported compute capabilities of this cuda version
    #   We export CUDA_CC so that cuda/Makefile knows what to include in the fat wrappers
    case $CUDAVERSION in
        "10.0") export CUDA_CC="50 52 53 60 61 62 70 72 75";; # Min. Linux driver: >=410.48
        "10.1") export CUDA_CC="50 52 53 60 61 62 70 72 75";; # Min. Linux driver: >=418.39
        "10.2") export CUDA_CC="50 52 53 60 61 62 70 72 75";; # Min. Linux driver: >=440.33
        "11.0") export CUDA_CC="50 52 53 60 61 62 70 72 75 80";; # Min. Linux driver: >=450.80.02
        "11.1") export CUDA_CC="50 52 53 60 61 62 70 72 75 80 86";; # Min. Linux driver: >=450.80.02 (Same CC for 11.1-11.7)
        "11.8") export CUDA_CC="50 52 53 60 61 62 70 72 75 80 86 87 89";; # Min. Linux driver: >=450.80.02
        "12.0") export CUDA_CC="50 52 53 60 61 62 70 72 75 80 86 87 89 90";; # Min. Linux driver: >=525.60.13 (Same CC for all 12.x.)
        "12.6") export CUDA_CC="50 52 53 60 61 62 70 72 75 80 86 87 89 90";; # Min. Linux driver: >=525.60.13 (Same CC for all 12.x.)
    esac

    # The path for shared libraries (relative to the build directory)
    RPATH=lib 
    mkdir -p $BUILDDIR/$RPATH
    
    # We overwrite the CGO Flags to make sure that it is compiled against $CUDAVERSION
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export CGO_LDFLAGS="-lcufft -lcurand -lcuda -L${CUDA_HOME}/lib64 -Wl,-rpath -Wl,\$ORIGIN/$RPATH"
    export CGO_CFLAGS="-I${CUDA_HOME}/include"

    # (Re)build everything
    (cd .. && make realclean && make -j 4 || exit 1)
    
    # Copy the executable and the cuda libraries to the output directory
    cp $GOPATH/bin/mumax3 $BUILDDIR 
    cp $GOPATH/bin/mumax3-convert $BUILDDIR 
    cp $GOPATH/bin/mumax3-server $BUILDDIR 
    cp $GOPATH/bin/mumax3-format $BUILDDIR 
    cp ../LICENSE $BUILDDIR
    cp $( ldd ${BUILDDIR}/mumax3 | grep libcufft | awk '{print $3}' ) ${BUILDDIR}/${RPATH}
    cp $( ldd ${BUILDDIR}/mumax3 | grep libcurand | awk '{print $3}' ) ${BUILDDIR}/${RPATH}

    (cd build && tar -czf ${MUMAX3UNAME}.tar.gz ${MUMAX3UNAME})

done
