{ pkgs ? import <nixpkgs> {} }:

with pkgs;

let
  virtualenvDir = "pythonenv";
  manylinuxLibPath = stdenv.lib.makeLibraryPath [(callPackage ./manylinux1.nix {}).package];
in
mkShell {
  buildInputs = [
    busybox
    git
    nodejs

    (python36Full.withPackages (pythonPkgs: with pythonPkgs; [
      tkinter
      virtualenvwrapper
    ]))
  ];

  # Fix wheel building and init virtualenv
  shellHook = ''
    unset SOURCE_DATE_EPOCH
    if [ ! -d "${virtualenvDir}" ]; then
      virtualenv --system-site-packages ${virtualenvDir}
    fi
    echo "manylinux1_compatible = True" > ${virtualenvDir}/lib/python3.6/_manylinux.py
    source ${virtualenvDir}/bin/activate
    export LD_LIBRARY_PATH=${manylinuxLibPath}
    export TMPDIR=/tmp
  '';
}
