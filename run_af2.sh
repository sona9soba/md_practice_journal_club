#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:rtx3050ti:1

export TF_FORCE_UNIFIED_MEMORY="1"
export XLA_PYTHON_CLIENT_MEM_FRACTION="2"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export TF_FORCE_GPU_ALLOW_GROWTH="true"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"


#ARGUMENTS
#parser.add_argument( "-pdbdir", type=str, default="", help='The name of a directory of pdbs to run through the model' )
#parser.add_argument( "-silent", type=str, default="", help='The name of a silent file to run through the model' )
#parser.add_argument( "-outpdbdir", type=str, default="outputs", help='The directory to which the output PDB files will be written. Only used when -pdbdir is active' )
#parser.add_argument( "-outsilent", type=str, default="out.silent", help='The name of the silent file to which output structs will be written. Only used when -silent is active' )
#parser.add_argument( "-runlist", type=str, default='', help="The path of a list of pdb tags to run. Only used when -pdbdir is active (default: ''; Run all PDBs)" )
#parser.add_argument( "-checkpoint_name", type=str, default='check.point', help="The name of a file where tags which have finished will be written (default: check.point)" )
#parser.add_argument( "-scorefilename", type=str, default='out.sc', help="The name of a file where scores will be written (default: out.sc)" )
#parser.add_argument( "-maintain_res_numbering", action="store_true", default=False, help='When active, the model will not renumber the residues when bad inputs are encountered (default: False)' )
#parser.add_argument( "-debug", action="store_true", default=False, help='When active, errors will cause the script to crash and the error message to be printed out (default: False)')
# AF2-Specific Arguments
#parser.add_argument( "-max_amide_dist", type=float, default=3.0, help='The maximum distance between an amide bond\'s carbon and nitrogen (default: 3.0)' )
#parser.add_argument( "-recycle", type=int, default=3, help='The number of AF2 recycles to perform (default: 3)' )
#parser.add_argument( "-no_initial_guess", action="store_true", default=False, help='When active, the model will not use an initial guess (default: False)' )
#parser.add_argument( "-force_monomer", action="store_true", default=False, help='When active, the model will predict the structure of a monomer (default: False)' )

AF2_DIR=/home/sona/dl_binder_design/af2_initial_guess
INPUT_DIR=$1
OUTPUT_DIR=$2
SCOREFILENAME=$3

python3 $AF2_DIR/predict.py \
    -pdbdir $INPUT_DIR \
    -outpdbdir $OUTPUT_DIR \
    -scorefilename $SCOREFILENAME \
    -debug

#bash run_af2.sh /home/sona/LigandMPNN/outputs/modified/backbones /home/sona/dl_binder_design/af2_initial_guess/outputs/modified modified.sc