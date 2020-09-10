# Contributed by @thomasjm as a gist:
# https://gist.github.com/thomasjm/b0e757213096527faa888cd18b07c182

{stdenv, pkgs}:

with stdenv.lib;
with pkgs;

rec {
  # To be manylinux1 compatible, we have to be able to link against any of the libraries below (PEP 513)
  # https://www.python.org/dev/peps/pep-0513
  desiredLibraries = [
    "libpanelw.so.5"
    "libncursesw.so.5"
    "libgcc_s.so.1"
    "libstdc++.so.6"
    "libm.so.6"
    "libdl.so.2"
    "librt.so.1"
    "libcrypt.so.1"
    "libc.so.6"
    "libnsl.so.1"
    "libutil.so.1"
    "libpthread.so.0"
    "libresolv.so.2"
    "libX11.so.6"
    "libXext.so.6"
    "libXrender.so.1"
    "libICE.so.6"
    "libSM.so.6"
    "libGL.so.1"
    "libgobject-2.0.so.0"
    "libgthread-2.0.so.0"
    "libglib-2.0.so.0"
  ];

  # The desired libraries can be collectively found in these packages
  libs = [glib libGL ncurses5 xorg.libX11 xorg.libXrender xorg.libXext xorg.libICE xorg.libSM glibc gcc7.cc];

  package = stdenv.mkDerivation {
    name = "manylinux1_libs";

    unpackPhase = "true";

    buildInputs = libs;
    propagatedBuildInputs = libs;

    buildPhase = ''
      mkdir -p $out/lib
      num_found=0
      IFS=:
      export DESIRED_LIBRARIES=${concatStringsSep ":" desiredLibraries}
      export LIBRARY_PATH=${makeLibraryPath libs}
      for desired in $DESIRED_LIBRARIES; do
        for path in $LIBRARY_PATH; do
          if [ -e $path/$desired ]; then
            echo "FOUND $path/$desired"
            ln -s $path/$desired $out/lib/$desired
            num_found=$((num_found+1))
            break
          fi
        done
      done
      num_desired=${toString (length desiredLibraries)}
      echo "Found $num_found of $num_desired libraries"
      if [ "$num_found" -ne "$num_desired" ]; then
        echo "Error: not all desired libraries were found"
        exit 1
      fi
    '';

    installPhase = "true";
  };
}
