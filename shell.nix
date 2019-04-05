{ pkgs ? import <nixpkgs> {} }:
let
  virtualenvDir = "pythonenv";
  liblapackShared = pkgs.liblapack.override { shared = true; };
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    busybox
    git
    pkgconfig
    hdf5
    libzip
    libpng
    freetype
    gfortran
    liblapackShared
    nodejs
    
    (python36.withPackages (pythonPkgs: with pythonPkgs; [
      virtualenvwrapper
    ]))
  ];

  # Fix wheel building and init virtualenv
  shellHook = ''
    unset SOURCE_DATE_EPOCH
    if [ ! -d "${virtualenvDir}" ]; then
      virtualenv ${virtualenvDir}
    fi
    source ${virtualenvDir}/bin/activate
    export TMPDIR=/tmp
    export LAPACK=${liblapackShared}/lib/liblapack.so
    export BLAS=${liblapackShared}/lib/libblas.so
  '';
}
