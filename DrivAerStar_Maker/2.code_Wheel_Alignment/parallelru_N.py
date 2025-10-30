
import pyvista as pv


base_path = "/work1/your_path/starccm/DrivAerStar_24000/base_STL/"
back = "Notchback/"
file_path = base_path+back+"/body/part_02_UB_EngineBayFlow.stl"
paras = r"/work1/your_path/starccm/DrivAerStar_24000/sldy/lhs_parameters_Notch_v3.csv"
out_path = r"/work1/your_path/starccm/DrivAerStar_24000/stl_N/"
input_path = "/work1/your_path/starccm/DrivAerStar_24000/stl_N/"


mesh = pv.read(file_path)
import numpy as np
cell_data = np.array(mesh.cell_centers().points)





Wheels_Front = base_path+back+"/part_05_Wheels_Front.stl"
Wheels_Front = pv.read(Wheels_Front)
Wheels_Front_points = np.array(Wheels_Front.points)
Wheels_Front_cell_data = np.array(Wheels_Front.cell_centers().points)

Wheels_Rear = base_path+back+"/part_06_Wheels_Rear.stl"
Wheels_Rear = pv.read(Wheels_Rear)
Wheels_Rear_cell_data = np.array(Wheels_Rear.cell_centers().points)

    


Wheels_Front_cell_data.max(axis=0),Wheels_Front_cell_data.min(axis=0)


Wheels_Front_left = Wheels_Front_cell_data[Wheels_Front_cell_data[:, 1] < 0]
(Wheels_Front_left.min(axis=0),Wheels_Front_left.max(axis=0))


Wheels_Front_left.max(axis=0),Wheels_Front_left.min(axis=0)


Wheels_Rear_left = Wheels_Rear_cell_data[Wheels_Rear_cell_data[:, 1] < 0]
(Wheels_Rear_left.min(axis=0),Wheels_Rear_left.max(axis=0))


(Wheels_Rear_left.min(axis=0)+Wheels_Rear_left.max(axis=0))/2


import numpy as np




mesh = pv.read(file_path)

def find_nearest_points(cell_data, target_point):
    
    diff_x = np.abs(cell_data[:, 0] - target_point[0])
    diff_y = np.abs(cell_data[:, 1] - target_point[1])
    diff_z = np.abs(cell_data[:, 2] - target_point[2])

    
    nearest_x_indices = np.argsort(diff_x)[:1000]
    nearest_y_indices = np.argsort(diff_y)[:1000]
    nearest_z_indices = np.argsort(diff_z)[:1000]

    def get_nearest_10(indices):
        points = cell_data[indices]
        distances = np.linalg.norm(points - target_point, axis=1)
        nearest_10_indices_in_subset = np.argsort(distances)[:100]
        nearest_10_indices_in_original = indices[nearest_10_indices_in_subset]
        nearest_10_points = points[nearest_10_indices_in_subset]
        return nearest_10_points, nearest_10_indices_in_original

    
    nearest_x_10, nearest_x_10_ids = get_nearest_10(nearest_x_indices)

    
    nearest_y_10, nearest_y_10_ids = get_nearest_10(nearest_y_indices)

    
    nearest_z_10, nearest_z_10_ids = get_nearest_10(nearest_z_indices)

    return (nearest_x_10, nearest_x_10_ids), (nearest_y_10, nearest_y_10_ids), (nearest_z_10, nearest_z_10_ids)

num_cells = cell_data.shape[0]
target_points=[1,1,1,1]
target_points[0] = np.array([6.99931844e+00, 7.60253225e+02, 2.66520182e-03])
target_points[1] = np.array([6.99931844e+00, -7.60253225e+02, 2.66520182e-03])
target_points[2] = np.array([2793.18452962, 763.42969767,    0.        ])
target_points[3] = np.array([2793.18452962, -763.42969767,    0.        ])
names = ['Wheels_Front_left','Wheels_Front_right','Wheels_Rear_left','Wheels_Rear_right']

wdw = []
for index, target_point in  enumerate( target_points):
    (x_points, x_ids), (y_points, y_ids), (z_points, z_ids) = find_nearest_points(cell_data, target_point)

    print("x 方向定位 id：")
    x_points_dw = (x_points.max(axis=0)+x_points.min(axis=0))/2
    print(x_ids)
    
    scalar_array = np.zeros(num_cells)
    scalar_array[x_ids] = 1
    mesh.cell_data[names[index] + 'x'] =  scalar_array

    print("y 方向定位 id：")
    y_points_dw = (y_points.max(axis=0)+y_points.min(axis=0))/2
    print(y_ids)
    scalar_array = np.zeros(num_cells)
    scalar_array[y_ids] = 1
    mesh.cell_data[names[index] + 'y'] =  scalar_array


    print("z 方向定位 id：")
    z_points_dw = (z_points.max(axis=0)+z_points.min(axis=0))/2
    print(z_ids)
    scalar_array = np.zeros(num_cells)
    scalar_array[z_ids] = 1
    mesh.cell_data[names[index] + 'z'] =  scalar_array
    print(x_points_dw[0], y_points_dw[1],z_points_dw[2], target_point)
    wdw.append([x_ids,y_ids,z_ids])
    





def sldw(mesh):
    res = []
    for x_ids,y_ids,z_ids in wdw:
        x_points = mesh.cell_centers().points[x_ids]
        y_points = mesh.cell_centers().points[y_ids]
        z_points = mesh.cell_centers().points[z_ids]

        x_points_dw = (x_points.max(axis=0)+x_points.min(axis=0))/2
        y_points_dw = (y_points.max(axis=0)+y_points.min(axis=0))/2
        z_points_dw = (z_points.max(axis=0)+z_points.min(axis=0))/2

        res.append([x_points_dw[0], y_points_dw[1],z_points_dw[2]])
        
        
    mean_fx = (res[0][0] + res[1][0])/2
    res[0][0],res[1][0] = mean_fx, mean_fx
    
    mean_fx = (res[2][0] +res[3][0])/2
    res[2][0],res[3][0] = mean_fx, mean_fx
    
    mean_fy = (res[0][1] -res[1][1])/2
    res[0][1],res[1][1] = mean_fy, -mean_fy
    
    mean_fy = (res[2][1] -res[3][1])/2
    res[2][1],res[3][1] = mean_fy, -mean_fy
    
    min_fz = min(res[2][2],res[3][2],res[0][2],res[1][2])
    res[0][2],res[1][2],res[2][2],res[3][2] = min_fz, min_fz,min_fz, min_fz
    
    
    return res




import pandas as pd


lhs_parameters = pd.read_csv(paras)
size_f =  lhs_parameters['car_size']
tires_diameter, tires_width = lhs_parameters['tires_diameter'],lhs_parameters[ 'tires_width']



def transform_a_wheel_data(diameter, width, Wheels_Front_cell_data,mask,dwx,dwy,dwz,size_f):
    
    
    diameter = diameter*1000
    width = width*1000
    Wheels_Front_left = Wheels_Front_cell_data[mask]

    
    center = np.array([
        (Wheels_Front_left[:, 0].min() + Wheels_Front_left[:, 0].max()) / 2,
        (Wheels_Front_left[:, 1].min() + Wheels_Front_left[:, 1].max()) / 2,
        (Wheels_Front_left[:, 2].min() + Wheels_Front_left[:, 2].max()) / 2 
    ])
    R = Wheels_Front_left[:, 0].max() - Wheels_Front_left[:, 0].min() 
    W = Wheels_Front_left[:, 1].max() - Wheels_Front_left[:, 1].min() 


    scale_matrix_diameter = np.array([1+diameter/R, 1+width/W, 1+diameter/R])*size_f
    Wheels_Front_left_centered = (Wheels_Front_left - center)
    
    
    Wheels_Front_cell_data[mask] = Wheels_Front_left_centered* scale_matrix_diameter 

    
    Wheels_Front_cell_data[mask, 0] += (dwx)
    Wheels_Front_cell_data[mask, 1] += (dwy)
    Wheels_Front_cell_data[mask, 2] += (dwz)


def left_mask(Wheels_Front_cell_data):
    
    mask = Wheels_Front_cell_data[:, 1] < 0
    return mask

def right_mask(Wheels_Front_cell_data):
    
    mask = Wheels_Front_cell_data[:, 1] > 0
    return mask



import multiprocessing
from tqdm import tqdm
import pyvista as pv
import os

def create_folder(id_str):
    folder_path = os.path.join(out_path, id_str)
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"文件夹 {folder_path} 创建成功")
        else:
            print(f"文件夹 {folder_path} 已经存在")
    except FileExistsError:
        print(f"文件夹 {folder_path} 已经存在")
    except PermissionError:
        print(f"没有权限创建文件夹 {folder_path}")
    except Exception as e:
        print(f"创建文件夹 {folder_path} 时发生错误: {e}")

def process_item(i):
    id_str = f"{i:05d}"
    try:
        if os.path.exists(input_path+rf"{id_str}/floor.txt"):
            print(f"文件夹 {id_str} 已完成floor.txt，跳过")
            return
        dwmesh = pv.read(input_path+rf"{id_str}/part_02_UB_EngineBayFlow.stl")
        fr, fl, rr, rl = sldw(dwmesh)
        ds, dr, dw = size_f[i], tires_diameter[i], tires_width[i]

        create_folder(id_str)
        
        wheels_front_path = base_path+back+"/part_05_Wheels_Front.stl"
        wheels_front = pv.read(wheels_front_path)
        transform_a_wheel_data(dr, dw, wheels_front.points, left_mask(wheels_front.points), fl[0], fl[1], fl[2], ds)
        transform_a_wheel_data(dr, dw, wheels_front.points, right_mask(wheels_front.points), fr[0], fr[1], fr[2], ds)
        output_front_path = input_path+rf"/{id_str}/part_05_Wheels_Front.stl"
        wheels_front.save(output_front_path)

        wheels_rear_path = base_path+back+"/part_05_Wheels_Front.stl"
        wheels_rear = pv.read(wheels_rear_path)
        transform_a_wheel_data(dr, dw, wheels_rear.points, left_mask(wheels_rear.points), rl[0], rl[1], rl[2], ds)
        transform_a_wheel_data(dr, dw, wheels_rear.points, right_mask(wheels_rear.points), rr[0], rr[1], rr[2], ds)
        output_rear_path =input_path+ rf"/{id_str}/part_06_Wheels_Rear.stl"
        wheels_rear.save(output_rear_path)

        min_z =  wheels_rear.points[:, 2].min()
        with open(input_path+rf"/{id_str}/floor.txt", 'w') as f:
            f.write(f"{min_z}\n")
    except Exception as e:
        print(f"Error processing item {id_str}: {e}")


if __name__ == "__main__":
    with multiprocessing.Pool(30) as pool:
        items = range(0, 8000)
        results = list(tqdm(pool.imap(process_item, items), total=len(items)))

    
