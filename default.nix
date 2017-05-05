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
        python35Packages.numpy
        python35Packages.scipy
        python35Packages.scikitlearn
        python35Packages.seaborn
        python35Packages.jupyter_core
        python35Packages.jupyter
        python35Packages.jupyter_client
        python35Packages.jupyterlab
        python35Packages.backports_shutil_get_terminal_size
        python35Packages.ipython
        python35Packages.ipykernel
	python35Packages.tensorflow
	python35Packages.networkx
	python35Packages.h5py
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
