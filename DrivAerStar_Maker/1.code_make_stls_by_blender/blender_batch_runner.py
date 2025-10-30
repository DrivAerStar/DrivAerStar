import os
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime

def run_blender_task(start, end):
    """run Blender task"""
    blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"
    script_path = os.path.abspath("make.blender.py")
    log_dir = Path(__file__).parent / "blender_jobs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"job_{start}_{end}.log"
    
    cmd = [
        blender_exe,
        "--background",
        "--python", script_path,
        str(start), str(end)
    ]
    
    print(f"run: {start}-{end} (log: {log_file})")
    
    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,  
            encoding="utf-8",         
            errors="replace"          
        )
        
        for line in process.stdout:
            f.write(line)
        
        process.wait()
    
    return start, end, process.returncode

def main():
    print("=== Blender===")
    print(f"run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    batch_size = 500
    total_tasks = 16
    max_concurrent = 16  
    
    tasks = [(i * batch_size, (i + 1) * batch_size) for i in range(total_tasks)]
    
    with multiprocessing.Pool(processes=max_concurrent) as pool:
        results = []
        
        for task in tasks:
            result = pool.apply_async(run_blender_task, task)
            results.append(result)
        
        pool.close()
        pool.join()
    
    success_count = 0
    failed_tasks = []
    
    for result in results:
        start, end, returncode = result.get()
        if returncode == 0:
            success_count += 1
        else:
            failed_tasks.append((start, end))
    
    print("\n=== info ===")
    print(f"all: {total_tasks}")
    print(f"win: {success_count}")
    print(f"error: {len(failed_tasks)}")
    
    if failed_tasks:
        print("\n error tasks:")
        for start, end in failed_tasks:
            print(f"  - {start}-{end}")
    
    print(f"end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()