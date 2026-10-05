#!/bin/bash
mkdir -p "${DG_DATA:-/workspace/_deps/data}"
cd "${DG_DATA:-/workspace/_deps/data}"
dl () { echo "START $1"; timeout 1800 curl -sL -o "$2" "$1" && echo "OK $2 $(stat -c%s "$2")" || echo "FAIL $2"; }
dl "https://huggingface.co/datasets/yairschiff/qm9/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet" qm9.parquet &
dl "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip" arxiv.zip &
dl "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000005640&format=fasta&compressed=true" human_proteome.fasta.gz &
dl "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" coco_ann.zip &
dl "https://www.openslr.org/resources/12/dev-clean.tar.gz" libri_devclean.tar.gz &
wait
echo ALLDONE
