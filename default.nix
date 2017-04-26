with import <nixpkgs> {};

python36Packages.buildPythonPackage {
    name = "thesis-env";
    buildInputs = [
        git
        libzip
	libpng
	freetype
        python36Full
        python36Packages.virtualenv
 	python36Packages.setuptools
	python36Packages.matplotlib
        python36Packages.Keras
	python36Packages.pandas
        python36Packages.numpy
        python36Packages.scipy
        python36Packages.scikitlearn
        python36Packages.seaborn
        python36Packages.jupyter_core
        python36Packages.jupyter
        python36Packages.jupyter_client
        python36Packages.jupyterlab
        python36Packages.backports_shutil_get_terminal_size
        python36Packages.ipython
        python36Packages.ipykernel
	#python35Packages.tensorflow
	#python36Packages.networkx
	python36Packages.h5py
        stdenv
        zlib
	bazel
        graphviz
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
