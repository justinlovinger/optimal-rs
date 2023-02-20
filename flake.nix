{
  inputs = {
    naersk = {
      url = "github:nix-community/naersk/master";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
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
        doCheck = true;
      };
      devShell = with pkgs;
        mkShell {
          buildInputs = [
            cargo
            cargo-tarpaulin
            rust-analyzer
            rustc
            rustfmt
            rustPackages.clippy
          ];
        };
    });
}
