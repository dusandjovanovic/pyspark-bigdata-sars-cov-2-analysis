#!/usr/bin/env bash

if [ -x "$(which pipenv)" ]
then
    if [ ! -e Pipfile.lock ]
    then
        echo 'ERROR: Pipfile missing'
        exit 1
    fi

    pipenv lock -r > dependencies.txt

    touch dependencies.txt
    pip3 install -r dependencies.txt --target ./packages

    if [ -z "$(ls -A packages)" ]
    then
        touch packages/empty.txt
    fi

    if [ ! -d packages ]
    then
        echo 'ERROR: Pip failed to import'
        exit 1
    fi

    cd packages
    zip -9mrv packages.zip .
    mv packages.zip ..
    cd ..

    rm -rf packages
    rm requirements.txt

    zip -ru9 packages.zip dependencies -x dependencies/__pycache__/\*

    exit 0
else
    echo 'ERROR: Pip is not installed'
    exit 1
fi