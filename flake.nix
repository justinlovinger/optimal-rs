{
  inputs = {
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    naersk = {
      url = "github:nix-community/naersk/master";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    fenix,
    nixpkgs,
    utils,
    naersk,
  }:
    utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {inherit system;};
      naersk-lib = pkgs.callPackage naersk {};
    in {
      defaultPackage = naersk-lib.buildPackage {
        src = ./.;
        doCheck = false; # Tests require nightly Rust.
      };
      devShell = with pkgs;
        mkShell {
          nativeBuildInputs = [
            cargo-readme
            fenix.packages.${system}.latest.toolchain
          ];

          shellHook = ''
            ln -f .hooks/* .git/hooks/
          '';
        };
    });
}
