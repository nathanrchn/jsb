#!/bin/bash

if [ ! -d "/jsb" ]; then
  git clone https://github.com/nathanrchn/jsb.git /jsb
else
  cd /jsb && git pull
fi

conda activate default
