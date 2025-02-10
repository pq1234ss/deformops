# This Nix expression defines a function that creates a FHS environment
# with Micromamba, and loads a specified Conda environment from a YAML file.
pkgs: name:
let
  # Path to the Micrimamba environment file.
  file = ./${name}.yaml;
in
pkgs.buildFHSEnv {
  name = "micromamba-${name}";
  targetPkgs = ps: [
    ps.uv
    ps.micromamba
    ps.git
    ps.coreutils
    ps.bashInteractive
    ps.jq # used to patch the state file
    ps.zlib # numpy requires zlib
  ];
  profile =
    let
      prefix = "$MAMBA_ROOT_PREFIX/envs/${name}";
    in
    ''
      set -e

      export MAMBA_ROOT_PREFIX="$(pwd)/.mamba"

      # Micromamba
      # ----------
      #
      # Configure micromamba to use a project-local directory.
      eval "$(micromamba shell hook --shell posix)"

      # Create the conda environment from file if it doesn't exist.
      if [ ! -d "${prefix}" ]; then
        echo "Conda environment not found. Creating from ${file}..."
        micromamba create --prefix "${prefix}" --file ${file} --yes

        CONDA_PREFIX=${prefix} ${./micromamba-patch.sh}

        micromamba activate "${prefix}"
        echo "Micromamba environment created and activated."

        pip install -e .[dev,test]
      else
        # Activate the environment.
        micromamba activate "${prefix}"
        echo "Micromamba environment activated."
      fi


      set +e
    '';
}
