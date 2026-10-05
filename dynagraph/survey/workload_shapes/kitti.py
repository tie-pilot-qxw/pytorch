import os,sys,glob
W=os.environ.get('DG_DATA','/workspace/_deps/data'); sys.path.insert(0,W+'/pylibs')
import numpy as np
fs=sorted(glob.glob(W+'/kitti/**/velodyne_points/data/*.bin',recursive=True))
print("KITTI raw drive 2011_09_26_drive_0093, %d LiDAR frames (HDL-64E)"%len(fs))
npts=[]; V={}
# SECOND / VoxelNet KITTI config: range x[0,70.4] y[-40,40] z[-3,1], voxel (0.05,0.05,0.1)
lo=np.array([0,-40,-3.0]); hi=np.array([70.4,40,1.0]); vs=np.array([0.05,0.05,0.1])
stages={1:[],2:[],4:[],8:[]}
for f in fs:
    p=np.fromfile(f,dtype=np.float32).reshape(-1,4)
    npts.append(len(p))
    xyz=p[:,:3]
    m=np.all((xyz>=lo)&(xyz<hi),axis=1)
    idx=((xyz[m]-lo)/vs).astype(np.int32)
    for s in stages:
        q=idx//np.array([s,s,s if s>1 else 1])
        key=(q[:,0].astype(np.int64)<<40)|(q[:,1].astype(np.int64)<<20)|q[:,2].astype(np.int64)
        stages[s].append(len(np.unique(key)))
npts=np.array(npts)
print("points/frame: min %d max %d mean %.0f distinct=%d  (over %d frames)"%(npts.min(),npts.max(),npts.mean(),len(set(npts.tolist())),len(npts)))
print("SECOND-style voxelization, voxel (0.05,0.05,0.1) m, KITTI range:")
tup=[]
for s in sorted(stages):
    a=np.array(stages[s])
    print("  sparse-conv stage stride %d: active voxels min %d max %d mean %.0f distinct=%d"%(s,a.min(),a.max(),a.mean(),len(set(a.tolist()))))
tup=list(zip(*[stages[s] for s in sorted(stages)]))
print("  distinct 4-stage active-voxel TUPLES over %d frames: %d"%(len(fs),len(set(tup))))
# how many distinct if you cap at max_voxels=40000 (OpenPCDet test setting)
a=np.array(stages[1])
print("  frames exceeding max_voxels=40000 (OpenPCDet KITTI test cap): %d/%d"%(int((a>40000).sum()),len(a)))
