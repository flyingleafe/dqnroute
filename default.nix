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
        jupyter
        jupyter_client
        jupyter_console
        jupyter_core
        jupyterlab
        jupyterlab_launcher
        backports_shutil_get_terminal_size
        widgetsnbextension
        ipython
        ipykernel
        ipywidgets
        tensorflow
        thespian
        typing
        networkx
        h5py
        pyyaml
        tqdm
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
