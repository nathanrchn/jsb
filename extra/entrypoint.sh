#!/bin/bash

. /opt/conda/etc/profile.d/conda.sh

if [ ! -d "/home/jsb" ]; then
  git clone https://github.com/nathanrchn/jsb.git /home/jsb
else
  cd /home/jsb && git pull
fi

conda activate default
