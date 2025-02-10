{ pkgs, ... }:
let
  # Attribute set of PyTorch images with different CUDA versions.
  baseImages = builtins.listToAttrs (
    builtins.map
      (spec: {
        name = "${spec.finalImageName}:${spec.finalImageTag}";
        value = pkgs.dockerTools.pullImage spec;
      })
      [
        # Base images defined here.
        # See also: `./oci-search.sh` and `./oci-prefetch.sh`.
        {
          imageName = "pytorch/pytorch";
          imageDigest = "sha256:cf5aa3f7045a68c10d80f546746591c5ccae6a33729e5e32625ff76bd2c036fe";
          hash = "sha256-qiwnTb4G7IN45IDiiCgeEeqLJmNPrJxfL0RWZVJpkGQ=";
          finalImageName = "pytorch/pytorch";
          finalImageTag = "2.8.0-cuda12.9-cudnn9-devel";
        }
      ]
  );

  # Function to build a Docker image with our package installed on top of
  # a base image (e.g. from PyTorch's Docker Hub containers with CUDA).
  buildImage =
    base:
    pkgs.dockerTools.buildImage {
      name = "khwstolle/deformops";
      fromImage = base;

      runAsRoot = ''
        #!${pkgs.stdenv.shell}
        export PATH=/bin:/usr/bin:/sbin:/usr/sbin:$PATH
        groupadd -r deformops 
        useradd -r -g deformops -d /home/deformops -M deformops 
        mkdir -p /home/deformops
        chown deformops:deformops /home/deformops
      '';

      config = {
        Cmd = [
          "${pkgs.stdenv.shell}"
        ];
        WorkingDir = "/home/deformops";
        #Volumes = {
        #  "/data" = { };
        #};
      };
    };
in
{
  "2.8.0-cuda12.9-cudnn-9" = buildImage baseImages."pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel";
}
