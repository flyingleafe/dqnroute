with import <nixpkgs> {};
with pkgs.python35Packages;

buildPythonPackage {
    name = "thesis-env";
    buildInputs = [
        git
	emacs
        libzip
	libpng
	freetype
        python35Full
        python35Packages.virtualenv
 	python35Packages.setuptools
	python35Packages.matplotlib
        python35Packages.Keras
	python35Packages.pandas
        stdenv
        zlib
	bazel
    ];
    # When used as `nix-shell --pure`
    shellHook = ''
    unset http_proxy
    export GIT_SSL_CAINFO=/etc/ssl/certs/ca-bundle.crt
    '';
    # used when building environments
    extraCmds = ''
    unset http_proxy # otherwise downloads will fail ("nodtd.invalid")
    export GIT_SSL_CAINFO=/etc/ssl/certs/ca-bundle.crt
    '';
}
