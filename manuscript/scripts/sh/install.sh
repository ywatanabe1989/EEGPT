#!/bin/bash

install_tex_from_package_manager() {
    # Check for apt, yum, and dnf, and install packages using the available package manager
    if command -v apt > /dev/null; then
        install_tex_with_apt
    elif command -v yum > /dev/null; then
        install_tex_with_yum_dnf yum
    elif command -v dnf > /dev/null; then
        install_tex_with_yum_dnf dnf
    else
        echo "No suitable package manager found (apt, yum, or dnf). Exiting installation."
        exit 1
    fi
    }

install_tex_with_apt() {
    sudo apt update
    sudo apt-get install texlive-full -y
    sudo apt-get install tree -y
    sudo apt-get install bibtex -y
}

install_tex_with_yum_dnf() {
    local pkg_manager="$1"
    sudo "$pkg_manager" update -y
    sudo "$pkg_manager" makecache --refresh    
    sudo "$pkg_manager" install 'texlive*' -y
    sudo "$pkg_manager" install tree -y
    sudo "$pkg_manager" install epel-release -y
    sudo "$pkg_manager" install ImageMagick -y
}

# install_tex_from_source() {
#     cd /tmp
#     wget https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2023/texlive-20230313-source.tar.xz
#     tar xvf texlive-20230313-source.tar.xz

#     sudo yum install perl-Digest-MD5 libXaw-devel -y

#     mkdir texlive-build
#     cd texlive-build
#     ../texlive-20230313-source/configure

#     make -j30
#     sudo make install -j30
# }

install_texcount() {
    wget https://mirrors.ctan.org/support/texcount.zip
    unzip texcount.zip
    sudo mv texcount /usr/share/texlive/texmf-dist/scripts/
    sudo chmod +x /usr/share/texlive/texmf-dist/scripts/texcount/texcount.pl
    sudo ln -s /usr/share/texlive/texmf-dist/scripts/texcount/texcount.pl /usr/bin/texcount
    rm texcount -rf
}

# install_elsarticle() {
#     wget https://hess.ess.washington.edu/repository/BCO_neon_2019/tex/elsarticle.cls
#     distdir='/usr/share/texlive/texmf-dist/tex/latex/elsarticle/'
#     sudo mkdir -p "$distdir"
#     sudo mv elsarticle.cls "$distdir"

#     sudo texhash $distdir
#     # echo  export TEXMFHOME="/home/ywatanabe/texmf" >> ~/.bashrc
#     # texhash /usr/share/texlive/texmf-dist/
#     # rm elsarticle.cls -rf    
#     }

# install_elsarticle() {
#     wget https://hess.ess.washington.edu/repository/BCO_neon_2019/tex/elsarticle.cls
#     distdir="$HOME/texmf/tex/latex/elsarticle/"
#     mkdir -p $distdir
#     mv elsarticle.cls "$distdir"
#     mktexlsr ~/texmf
#     texhash /$distdir
#     # rm elsarticle.cls -rf

    
#     }
install_elsarticle() {
    # Remove the now-empty elsarticle directory and the zip file
    rmdir elsarticle
    rm -f elsarticle.zip
    
    # Download the elsarticle package from CTAN
    wget https://mirrors.ctan.org/macros/latex/contrib/elsarticle.zip
    
    # Unzip the downloaded package
    unzip elsarticle.zip

    # Navigate to the elsarticle directory
    cd elsarticle
    
    # Run LaTeX on the .ins file to generate the .cls file
    latex elsarticle.ins

    # Go back to the previous directory
    cd ..
    
    # Define the directory where LaTeX packages are stored for the current user
    distdir="$HOME/texmf/tex/latex/"
    
    # Remove any previous elsarticle installations in the directory
    rm -rf "${distdir}elsarticle"
    
    # Create the directory if it doesn't exist
    mkdir -p "${distdir}elsarticle"
    
    # Move the elsarticle package files to the user's LaTeX directory
    mv elsarticle/* "${distdir}elsarticle/"
    
    # Remove the now-empty elsarticle directory and the zip file
    rmdir elsarticle
    rm -f elsarticle.zip
    
    # Update the TeX filename database
    mktexlsr "$HOME/texmf"

    mkdir -p $HOME/texmf/bibtex/bst/elsarticle
    mv $HOME/texmf/tex/latex/elsarticle/*.bst $HOME/texmf/bibtex/bst/elsarticle/
    
    # No need to run mktexlsr or texhash on the subdirectory, as it's included in the parent directory update
    # List the files to confirm they've been installed
    ls "${distdir}elsarticle"
}

# install_elsarticle() {
#     wget https://mirrors.ctan.org/macros/latex/contrib/elsarticle.zip
#     unzip elsarticle.zip
#     distdir="$HOME/texmf/tex/latex/"
#     rm $distdir -rf
#     mkdir -p $distdir
#     mv elsarticle "$distdir"
#     mktexlsr ~/texmf
#     mktexlsr ~/texmf/tex/latex/elsarticle
#     texhash ~/texmf
#     texhash ~/texmf/tex/latex/elsarticle
#     ls ~/texmf/tex/latex/elsarticle        
#     rm elsarticle.zip -f
# }



install_texcount() {
    wget https://mirror.cse.unsw.edu.au/pub/CTAN/macros/latex/contrib/wordcount/wordcount.sh
    wget https://mirror.cse.unsw.edu.au/pub/CTAN/macros/latex/contrib/wordcount/wordcount.tex
    
    chmod +x wordcount.sh
    mv wordcount.sh texcount
    sudo mv texcount /usr/local/bin
    }


generate_python_environment() {
    python -m venv env
    ./env/bin/python -m pip install -U pip
    ./env/bin/python -m pip install -r ./scripts/py/requirements.txt
    }

main() {
    install_tex_from_package_manager # Ubuntu is okay with this

    # Centos/Rocky Linux
    # install_tex_from_souce
    install_elsarticle
    # install_texcount

    generate_python_environment
}

main

## EOF
