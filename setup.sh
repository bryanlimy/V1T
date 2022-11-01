#!/bin/sh

device=false
current_dir="$(pwd -P)"

check_requirements() {
    case $(uname -s) in
        Darwin)
            if [ "$(uname -m)" = "arm64" ]; then
                printf "macOS (Apple Silicon) system detected.\n"
                device="osx-arm64"
            else
                printf "macOS (Intel) system detected.\n"
                export CFLAGS='-stdlib=libc++'
                device="osx-64"
            fi
            ;;
        Linux)
            printf "Linux system detected.\n"
            device="linux-64"
            ;;
        *)
            printf "Only Linux and macOS are currently supported.\n"
            exit 1
            ;;
    esac
}

install_torch() {
    printf "\nInstalling PyTorch...\n"
    case $device in
        osx-arm64)
            conda install -c pytorch pytorch torchvision torchaudio -y
            ;;
        osx-64)
            conda install -c pytorch pytorch torchvision torchaudio -y
            ;;
        linux-64)
            conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
            ;;
        *)
            pip install pytorch torchvision torchaudio
            ;;
    esac
}

install_packages() {
    printf "\nInstall Python packages...\n"
    pip install -e .
}

check_requirements
install_torch
install_packages

printf '\n\nSetup completed.\n'
