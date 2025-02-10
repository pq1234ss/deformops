# Support for various CUDA versions.
# See https://github.com/NixOS/nixpkgs/blob/nixos-unstable/doc/languages-frameworks/cuda.section.md

{
  description = "Deformable grid sampling operations.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nix-gl-host.url = "github:numtide/nix-gl-host";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    inputs@{ self, flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
      ];
      systems = [
        "x86_64-linux"
      ];
      perSystem =
        {
          config,
          system,
          ...
        }:
        let
          # Nixpkgs package set with CUDA support enabled.
          pkgs = import inputs.nixpkgs {
            inherit system;
            config = {
              #cudaCapabilities = [ <target-architectures> ];
              cudaForwardCompat = true;
              cudaSupport = true;
              allowUnfree = true;
            };
          };

          # OCI images
          ociImages = import ./environments/oci.nix { inherit inputs pkgs; };
          ociPackages = pkgs.lib.mapAttrs' (name: value: {
            name = "oci-${name}";
            value = value;
          }) ociImages;

          # UV
          uvBuild = import ./environments/uv.nix;
          uvFHS = [
            (uvBuild {
              inherit inputs;
              pkgs = pkgs.cudaPackages_12_6.pkgs;
              name = "cu126";
            })
            (uvBuild {
              inherit inputs;
              pkgs = pkgs.cudaPackages_12_8.pkgs;
              name = "cu128";
            })
          ];
          uvShells = builtins.listToAttrs (
            builtins.map (fhs: {
              name = fhs.name;
              value = fhs.env;
            }) uvFHS
          );

          # Micromamba
          mmBuild = import ./environments/micromamba.nix pkgs;
          mmFHS = builtins.map mmBuild [
            "py313cu128"
            "py313cu129"
          ];
          mmShells = builtins.listToAttrs (
            builtins.map (fhs: {
              name = fhs.name;
              value = fhs.env;
            }) mmFHS
          );
        in
        {
          packages = ociPackages;
          devShells = {
            default = self.devShells.${system}.uv-cu128;
          }
          // uvShells
          // mmShells;
          formatter = pkgs.alejandra;
        };
      flake = { };
    };
}
