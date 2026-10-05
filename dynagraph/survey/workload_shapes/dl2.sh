#!/bin/bash
W="${DG_DATA:-/workspace/_deps/data}"; mkdir -p "$W"
dl () { echo "START $2"; timeout 1800 curl -sL -o "$W/$2" "$1" && echo "OK $2 $(stat -c%s "$W/$2")" || echo "FAIL $2"; }
dl "http://www.quantum-machine.org/gdml/repo/datasets/md22_Ac-Ala3-NHMe.npz" md22_alaala.npz &
dl "http://www.quantum-machine.org/gdml/repo/datasets/md22_AT-AT-CG-CG.npz" md22_atatcgcg.npz &
dl "http://www.quantum-machine.org/gdml/repo/datasets/md22_buckyball-catcher.npz" md22_bucky.npz &
dl "http://www.quantum-machine.org/gdml/repo/datasets/md22_double-walled_nanotube.npz" md22_nanotube.npz &
dl "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0093/2011_09_26_drive_0093_sync.zip" kitti_drive.zip &
wait
echo ALLDONE2
