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
      naersk-lib = pkgs.callPackage naersk {
        cargo = toolchain;
        rustc = toolchain;
      };
      toolchain = fenix.packages.${system}.latest.toolchain;
    in {
      defaultPackage = naersk-lib.buildPackage {
        src = ./.;
        doCheck = true;
        cargoTestOptions = xs: xs ++ ["--all"];
      };
      devShell = with pkgs;
        mkShell {
          nativeBuildInputs = [
            # We should add `cargo-export` for benchmarking,
            # once it is available in Nixpkgs.
            cargo-edit
            cargo-readme
            toolchain
          ];

          shellHook = ''
            ln -f .hooks/* .git/hooks/
          '';

          PROPTEST_DISABLE_FAILURE_PERSISTENCE = 1;
        };
    });
}
