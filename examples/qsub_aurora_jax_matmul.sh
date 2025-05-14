#!/bin/bash -l
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -q gpu_hack_prio
#PBS -l filesystems=home:flare
#PBS -A gpu_hack
#PBS -N default
#PBS -o logs/
#PBS -e logs/

# Change directory to one where qsub was executed
cd ${PBS_O_WORKDIR}


#module unload oneapi/eng-compiler/2024.07.30.002
#module use /opt/aurora/24.180.3/spack/unified/0.8.0/install/modulefiles/oneapi/2024.07.30.002
#module use /soft/preview/pe/24.347.0-RC2/modulefiles
#module add oneapi/release
#module use /lus/flare/projects/datasets/softwares/frameworks_factory/aurora_fw_2025.0.1_u1_test_lus-umd1077p18/modulefiles
export FI_MR_CACHE_MONITOR=disabled
module load frameworks

#conda activate /lus/flare/projects/datasets/softwares/envs/jax_0p4p31

export LD_LIBRARY_PATH=/opt/aurora/24.347.0/oneapi/ccl/2021.14/lib:$LD_LIBRARY_PATH

export CPU_BIND="verbose,list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"
export CCL_WORKER_AFFINITY="42,43,44,45,46,47,94,95,96,97,98,99"
export ZE_AFFINITY_MASK="0,1,2,3,4,5,6,7,8,9,10,11"

# Variables for multinode
NNODES=1
NRANKS=12 # Number of MPI ranks to spawn per node

NTOTRANKS=$(( NNODES * NRANKS ))

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS}"

# run my code
mpiexec -n ${NTOTRANKS} -ppn ${NRANKS} --cpu-bind ${CPU_BIND} python jax_matmul.py 
