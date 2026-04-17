#!/bin/bash

curl https://sh.rustup.rs -sSf | RUSTUP_HOME=./.rustup CARGO_HOME=./.cargo sh -s -- -y --no-modify-path --default-toolchain stable --profile minimal 2> /dev/null
export PATH="$PWD/.cargo/bin:$PATH"
rustup install stable
make

echo "to be able to run build commands again through MAKE, source this  file"
echo -e  "\tsource installrustup.sh"
