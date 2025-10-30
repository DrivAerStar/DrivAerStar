base_case_path="/public/home/your_path/work/starccm/DrivAerStar_12000/stl_F/"
pvpython_path="/public/home/your_path/pkg/ParaView-5.13.3-MPI-Linux-Python3.10-x86_64/bin/pvpython"
max_jobs=30

for dir in "$base_case_path"/*/; do
    folder_name=$(basename "$dir")
    case_dir="$base_case_path/$folder_name/case/"
    case_file_path="$case_dir/KwWakeRefine0503.case"
    
    vtk_file_path="/work1/your_path/starccm/DrivAerStar_12000/vtk_F/$folder_name.vtk"

    if [ -f "$vtk_file_path" ]; then
        echo " $vtk_file_path skip"
        continue
    fi

    if [ ! -d "$case_dir" ]; then
        echo "case  $case_dir  skip"
        continue
    fi

    file_count=$(find "$case_dir" -type f | wc -l)
    if [ "$file_count" -ne 13 ]; then
        echo "skip $case_dir  file_count is not 13"
        continue
    fi
    
    (
        echo "run $pvpython_path 2F.pv.py $case_file_path $vtk_file_path"
        $pvpython_path 2F.pv.py "$case_file_path" "$vtk_file_path"
        exit_status=$?
        if [ $exit_status -ne 0 ]; then
            echo "run $pvpython_path 2.pv.py $case_file_path $vtk_file_path error exit_status: $exit_status"
        else
            echo "run $pvpython_path 2.pv.py $case_file_path $vtk_file_path win"
        fi
    ) &

    while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
        current_jobs=$(jobs -r | wc -l)
        echo "running: $current_jobs ..."
        sleep 1
    done
done


wait