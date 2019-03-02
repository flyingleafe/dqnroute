with import <nixpkgs> {};

python36Packages.buildPythonPackage {
    name = "dqnroute";
    src = ./src;
    buildInputs = with python36Packages; [
        # system packages
        git
        libzip
	libpng
	freetype
        
        # python packages
        python36Full
        virtualenv
 	setuptools
	matplotlib
        Keras
	pandas
        more-itertools
        numpy
        simpy
        scipy
        scikitlearn
        seaborn
        jupyter_core
        jupyter
        jupyter_client
        jupyterlab
        backports_shutil_get_terminal_size
        ipython
        ipykernel
	tensorflow
        thespian
	networkx
	h5py
	pyyaml
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
