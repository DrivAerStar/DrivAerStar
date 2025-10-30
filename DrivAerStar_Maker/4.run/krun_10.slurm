#!/bin/bash
#SBATCH -N 1
#SBATCH -p write_your_queue
#SBATCH --ntasks-per-node=30


NUM=$1
BASE_PATH="/work1/your_path/starccm/DrivAerStar_24000/stl_E"
STARCCM="/public/home/your_path/soft/star1806/18.06.007-R8/STAR-CCM+18.06.007-R8/star/bin/starccm+"

# run 10 by 10
for i in {0..9}; do
  FOLDER=$(printf "%05d" $((NUM * 10 + i)))
  FOLDER_PATH="${BASE_PATH}/${FOLDER}"
  
  echo "FOLDER ${FOLDER}"
  
  KEY_SCRIPT="${FOLDER_PATH}/KwWakeRefine0521.java"
  if [ ! -f "$KEY_SCRIPT" ]; then
    echo "skip ${FOLDER}"
    continue
  fi
  
  SCRIPT1="${FOLDER_PATH}/stl_2_dbs.java"
  LOG1="${FOLDER_PATH}/2dbs.log"
  if [ -f "$SCRIPT1" ]; then
    echo "run : stl_2_dbs.java"
    cd "$FOLDER_PATH" && $STARCCM -locale zh -np $SLURM_NTASKS -power -batch "$SCRIPT1" > "$LOG1" 2>&1
    if [ $? -ne 0 ]; then
      echo "error:  run stl_2_dbs "
      continue
    fi
  else
    echo "skip : file not exist"
  fi
  
  SCRIPT2="${FOLDER_PATH}/baomian.java"
  LOG2="${FOLDER_PATH}/baomian.log"
  if [ -f "$SCRIPT2" ]; then
    echo "run2: baomian00.java"
    cd "$FOLDER_PATH" && $STARCCM -np $SLURM_NTASKS -power -batch "$SCRIPT2" > "$LOG2" 2>&1
    if [ $? -ne 0 ]; then
      echo "error: skip else"
      continue
    fi
  else
    echo "skip: file not exist"
  fi
  
  SCRIPT3="${FOLDER_PATH}/KwWakeRefine0521.java"
  LOG3="${FOLDER_PATH}/KwWakeRefine0521.log"
  if [ -f "$SCRIPT3" ]; then
    echo "run3: KwWakeRefine0521.java"
    cd "$FOLDER_PATH" && $STARCCM -locale zh -np $SLURM_NTASKS -power -batch "$SCRIPT3" > "$LOG3" 2>&1
    cd "$FOLDER_PATH" && rm *.sim
    cd "$FOLDER_PATH" && rm *.sim~
    cd "$FOLDER_PATH" && rm *.stl
    cd "$FOLDER_PATH" && rm *.dbs
    if [ $? -ne 0 ]; then
      echo "error "
    fi
  else
    echo "error: file not exist"
  fi



  echo "done ${FOLDER}"
  echo "------------------------"
done

echo "all file done"