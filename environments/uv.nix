# This Nix expression defines a function that builds an FHS environment with
# UV (a Python project manager) and a Nix-provided CUDA installation.
{
  inputs,
  pkgs,
  name,
  backendName ? name,
  venvDir ? ".venvs/${name}",
}:
pkgs.buildFHSEnv {
  name = "uv-${name}";
  targetPkgs =
    pkgs:
    [
      inputs.nix-gl-host.packages.${pkgs.system}.default
    ]
    ++ (
      with pkgs;
      [
        python3
        cmake
        ninja
        zlib
        uv
      ]
      ++ (with python3.pkgs; [
        setuptools
        wheel
      ])
      ++ (with cudaPackages; [
        backendStdenv.cc
        cudatoolkit
        cuda_cudart
        #cuda_cupti
        #cuda_nvrtc
        #cuda_nvtx
        cudnn
        #libcublas
        #libcufft
        #libcurand
        #libcusolver
        #libcusparse
        #libnvjitlink
        nccl
      ])
    );

  profile = ''
    # UV configuration. 
    # See: https://docs.astral.sh/uv/reference/environment/
    # export UV_NO_BUILD_ISOLATION=true
    export UV_SYSTEM_PYTHON=true
    export UV_PROJECT_ENVIRONMENT=${venvDir}
    export UV_TORCH_BACKEND=auto
    # export UV_NO_MANAGED_PYTHON=true
    export UV_PYTHON_DOWNLOADS=never

    # C compiler.
    # export CC=${pkgs.gcc}/bin/gcc
    # export CXX=${pkgs.gcc}/bin/g++
    # export PATH=${pkgs.gcc}/bin:$PATH

    # Linker.
    export LD_LIBRARY_PATH=$(nixglhost -p):$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="${
      pkgs.lib.makeLibraryPath [
        pkgs.cudaPackages.cudatoolkit
        pkgs.cudaPackages.cudnn
      ]
    }:$LD_LIBRARY_PATH"
    # export LD_LIBRARY_PATH="${pkgs.gcc.cc.lib}/lib:$LD_LIBRARY_PATH"

    export LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.cudaPackages.cudatoolkit ]}:$LIBRARY_PATH"

    # CUDA.
    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
    export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}

    echo "* Running UV sync..."
    uv sync --extra ${backendName} --preview-features extra-build-dependencies

    echo "* Activating virtual environment..."
    source ${venvDir}/bin/activate

    # Point TORCH_EXTENSIONS_DIR to the virtual environment
    export TORCH_EXTENSIONS_DIR=${venvDir}/torch_extensions
    mkdir -p "$TORCH_EXTENSIONS_DIR"
  '';
}
