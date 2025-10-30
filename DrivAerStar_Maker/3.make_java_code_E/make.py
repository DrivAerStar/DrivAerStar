import os

back = "Estate"
rm_dir = r"/work1/your_path/starccm/DrivAerStar_24000/stl_XXX/"
template_files = [
    r"baomian.template.java",
    r"KwWakeRefine0521.template.java",
    r"stl_2_dbs.template.java",
]


for template_file in template_files:
    if not os.path.exists(template_file):
        print(f"file {template_file} not exists")
    else:
        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()

        for dir_name in range(0 , 8000):
            id = "%05d" % dir_name
            dir_name = "%05d" % dir_name
            dir_path = os.path.join(rm_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            gt_path = rm_dir+"/"+ dir_name
            new_content = template_content.replace("<<<dir>>>", gt_path)
            new_content = new_content.replace("<<<id>>>", id)
            new_content = new_content.replace("<<<back>>>", back)
            
            new_file_path = os.path.join(dir_path, os.path.basename(template_file).replace('.template', ''))
            try:
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"write {new_file_path}")
            except Exception as e:
                print(f"write {new_file_path} error: {e}")
        
