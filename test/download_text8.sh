#!/bin/bash -e

WORKSPACE="$(cd $(dirname $0); pwd)/testdata/"

if [ ! -e "${WORKSPACE}/text8" ]; then
  echo "Download text8 corpus"
  if hash wget 2>/dev/null; then
    wget -q --show-progress -P "${WORKSPACE}" http://mattmahoney.net/dc/text8.zip
  else
    curl --progress-bar -o "${WORKSPACE}/text8.zip" -O http://mattmahoney.net/dc/text8.zip
  fi

  echo "Unzip text8.zip"
  unzip "${WORKSPACE}/text8.zip" -d ${WORKSPACE}
  rm "${WORKSPACE}/text8.zip"
else
  echo "Text8 has been already downloaded"
fi
