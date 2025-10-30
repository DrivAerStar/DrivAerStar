import bpy
import os
import random
import numpy as np


import bpy
from mathutils import Vector


for obj in bpy.data.objects:
    if obj.type == 'MESH' or obj.type == 'CURVE' or obj.type == 'SURFACE' or obj.type == 'FONT' or obj.type == 'ARMATURE' or obj.type == 'LATTICE' or obj.type == 'EMPTY' or obj.type == 'CAMERA' or obj.type == 'LAMP' or obj.type == 'SPEAKER':
        bpy.data.objects.remove(obj)



stl_dir = r"D:\DrivAer\DrivAer_STLs\Myselection\Notchback\body"
export_dir = r"D:\stl_N"
input_params = np.loadtxt(r'D:\AutoRANS\step1.1 ffd\lhs_parameters_Notch_v3.csv', delimiter=',', skiprows=1)

os.makedirs(stl_dir, exist_ok=True)
os.makedirs(export_dir, exist_ok=True)

imported_objects = []

for file in os.listdir(stl_dir):
    if file.lower().endswith('.stl'):
        file_path = os.path.join(stl_dir, file)
        
        bpy.ops.import_mesh.stl(
            filepath=file_path,
            global_scale=1e-3,  
            use_scene_unit=True
        )
        

        imported_obj = bpy.context.selected_objects[0]
        base_name = os.path.splitext(file)[0]  
        imported_obj.name = base_name
        imported_objects.append(base_name)
        

        bpy.ops.object.transform_apply(
            location=True,
            rotation=True,
            scale=True
        )



lattice_data = bpy.data.lattices.new("LatticeData")
lattice_object = bpy.data.objects.new("MyLatticeObject", lattice_data)
scene = bpy.context.scene
scene.collection.objects.link(lattice_object)

lattice_object.location = (1.5, 0, 0.45)  
lattice_object.scale = (5, 2.2, 1.7)   


lattice_data.points_u = 32
lattice_data.points_v = 8
lattice_data.points_w = 8


points = []
for point in lattice_data.points:
    points.append([point.co_deform.x, point.co_deform.y, point.co_deform.z])

points = np.array(points)
x = points[:,0]
y = points[:,1]
z = points[:,2]


# 
origin_points = [[p.co_deform.x, p.co_deform.y, p.co_deform.z] for p in lattice_data.points]

def reset_points():
    for  j in range(len(lattice_data.points)) :
        point = lattice_data.points[j]
        point.co_deform.x = origin_points[j][0]
        point.co_deform.y = origin_points[j][1]
        point.co_deform.z = origin_points[j][2]
    bpy.context.view_layer.update()



mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

for obj in mesh_objects:
    modifier = obj.modifiers.new(name="LatticeDeform", type='LATTICE')
    modifier.object = lattice_object
    modifier.strength = 1.0


# index
x_min = x.min()
y_min = y.min()
z_min = z.min()

x_max = x.max()
y_max = y.max()
z_max = z.max()

x_len  = x_max - x_min
z_len  = z_max - z_min
y_len  = y_max - y_min
def ramp_angle(points, angle_z):
    

    index_1 = np.where(
          (z <= z_min + x_len* (2/7))
        & (x <=  x_min + x_len* (4/31) ))[0]
   
    ramp_angle_index = [index_1]
    center = x_min + x_len* (4/31) 
    
    move_z = np.tan(angle_z) *(x_len* (4/31))
    def f(x,y,z,a):
        dx = 0
        dy = 0
        
        pc = (x-center)/(x_min-center)
        dz = pc * move_z
        return dx,dy,dz
    
    for j in ramp_angle_index[0]:
        point = lattice_data.points[j]
        dx,dy,dz = f(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move_z)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz
   

def car_green_house_angle(points, move_z):

    #9. Car Green House Angle
    z_Roof = points[:, 2].min()+0.6*(points[:, 2].max() -points[:, 2].min())
    zmax = points[:, 2].max()
    ymean = points[:, 1].mean()
    yl = ymean - points[:, 1].max()
    head = ( points[:, 2] >= z_Roof)
    index = np.where( head )[0]
    gha_index = [index]


    def fx_x(x,y,z,a):
        tan_a = np.tan(a) 
        dx = 0
        dy = (z-z_Roof)/(zmax-z_Roof)* tan_a*(y-ymean)/yl
        dz = 0
        return dx,dy,dz
    
    for j in gha_index[0]:
        point = lattice_data.points[j]
        dx,dy,dz = fx_x(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move_z)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz


def trunklid_angle(points, angle_z):
    
   
    index_1 = np.where(
          (z >= z_max - z_len* (3/7))
        & (x >=  x_max - x_len* (8/31) ))[0]

    ramp_angle_index = [index_1]
    center = x_max - x_len* (8/31)
    
    move_z = - np.tan(angle_z) *(x_len* (8/31))*2
    def f(x,y,z,a):
        dx = 0
        dy = 0
        
        pc = (x-center)/(x_min-center)
        dz = pc * move_z
        return dx,dy,dz
    
    for j in ramp_angle_index[0]:
        point = lattice_data.points[j]
        dx,dy,dz = f(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move_z)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz

def diffusor_angle(points, angle_z):
    
    index_1 = np.where(
          (z <= z_min + x_len* (3/7))
        & (x >=  x_max - x_len* (6/31) ))[0]

    ramp_angle_index = [index_1]
    center = x_max - x_len* (6/31)
    
    move_z = np.tan(angle_z) *(x_len* (6/31))*1.5
    def f(x,y,z,a):
        dx = 0
        dy = 0
        
        pc = (x-center)/(x_len* (6/31))
        dz = pc * move_z
        return dx,dy,dz
    
    for j in ramp_angle_index[0]:
        point = lattice_data.points[j]
        dx,dy,dz = f(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move_z)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz


def front_bumper_length(points,move_x):
    index_1 = np.where( x <=  x_min + x_len* (4/31) )[0]
        
    center = x_min + x_len* (4/31) 
    move_x = move_x/5
    def f(x,y,z,a):
        pc = (center - x)/(x_len *(4/31))
        dx = - pc * move_x
        dy = 0
        dz = 0
        return dx,dy,dz
    
    for j in index_1:
        point = lattice_data.points[j]
        dx,dy,dz = f(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move_x)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz

def side_mirrors_x(points,move_x):
    index_1 = np.where( 
       ((x_min + x_len* (11/31) <=  x) & (x <=  x_min + x_len* (13/31)))
      & ((y == y_min) | (y == y_max))
      & ((z_min + z_len* (4/7) <=  z) & (z  <=  z_min + z_len* (6/7)))
     )[0]
    move_x = move_x/5
    def f(x,y,z,a):
         dx = move_x
         dy = 0
         dz = 0
         return dx,dy,dz
    
    for j in index_1:
        point = lattice_data.points[j]
        dx,dy,dz = f(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move_x)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz

def side_mirrors_z(points,move_z):
    index_1 = np.where( 
       ((x_min + x_len* (11/31) <=  x) & (x <=  x_min + x_len* (13/31)))
      & ((y == y_min) | (y == y_max))
      & ((z_min + z_len* (4/7) <=  z) & (z  <=  z_min + z_len* (6/7)))
     )[0]
    move_z = move_z/1.7
    def f(x,y,z,a):
         dx = 0
         dy = 0
         dz = move_z
         return dx,dy,dz
    
    for j in index_1:
        point = lattice_data.points[j]
        dx,dy,dz = f(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move_z)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz

def wind_screen_x(points,move_x):
    index_1 = np.where( 
        ((x_min + x_len* (10/31) <=  x) & (x <=  x_min + x_len* (14/31)))
      & ((y_min  <  y ) & (y < y_max ))
      & ((z_min + z_len* (5/7) <=  z) & (z  <=  z_min + z_len* (6.5/7)))
     )[0]
    move_x = move_x/5
    def f(x,y,z,a):
         dx = move_x
         dy = 0
         dz = 0
         return dx,dy,dz
    
    for j in index_1:
        point = lattice_data.points[j]
        dx,dy,dz = f(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move_x)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz

def wind_screen_z(points,move_z):
    index_1 = np.where( 
        ((x_min + x_len* (10/31) <=  x) & (x <=  x_min + x_len* (14/31)))
      & ((y_min  <  y ) & (y < y_max ))
      & ((z_min + z_len* (5/7) <=  z) & (z  <=  z_min + z_len* (6.5/7)))
     )[0]
    move_z = move_z/1.7
    def f(x,y,z,a):
         dx = 0
         dy = 0
         dz = move_z
         return dx,dy,dz
    
    for j in index_1:
        point = lattice_data.points[j]
        dx,dy,dz = f(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move_z)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz

def rear_window_x(points,move_x):
    index_1 = np.where( 
        ((x_min + x_len* (15/31) <=  x) & (x <=  x_min + x_len* (27/31)))
      & ((y_min  <  y ) & (y < y_max ))
      & ((z_min + z_len* (5/7) <=  z) & (z  <=  z_min + z_len* (6.5/7)))
     )[0]
    move_x = move_x/5
    def f(x,y,z,a):
         dx = move_x
         dy = 0
         dz = 0
         return dx,dy,dz
    
    for j in index_1:
        point = lattice_data.points[j]
        dx,dy,dz = f(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move_x)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz
        
def rear_window_z(points,move_z):
    index_1 = np.where( 
        ((x_min + x_len* (15/31) <=  x) & (x <=  x_min + x_len* (27/31)))
      & ((y_min  <  y ) & (y < y_max ))
      & ((z_min + z_len* (5/7) <=  z) & (z  <=  z_min + z_len* (6.5/7)))
     )[0]
    move_z = move_z/1.7
    def f(x,y,z,a):
         dx = 0
         dy = 0
         dz = move_z
         return dx,dy,dz
    
    for j in index_1:
        point = lattice_data.points[j]
        dx,dy,dz = f(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move_z)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz


def trunklid_x(points,move_x):
    index_1 = np.where( 
        ((x_min + x_len* (27/31) <=  x) & (x <=  x_min + x_len* (30/31)))
      & ((y_min  <  y ) & (y < y_max ))
      & ((z_min + z_len* (1/7) <=  z) & (z  <=  z_min + z_len* (6/7)))
     )[0]
    move_x = move_x/5
    def f(x,y,z,a):
         dx = move_x
         dy = 0
         dz = 0
         return dx,dy,dz
    
    for j in index_1:
        point = lattice_data.points[j]
        dx,dy,dz = f(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move_x)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz
        
def trunklid_z(points,move_z):
    index_1 = np.where( 
        ((x_min + x_len* (27/31) <=  x) & (x <=  x_min + x_len* (30/31)))
      & ((y_min  <  y ) & (y < y_max ))
      & ((z_min + z_len* (1/7) <=  z) & (z  <=  z_min + z_len* (6/7)))
     )[0]
    move_z = move_z/1.7
    def f(x,y,z,a):
         dx = 0
         dy = 0
         dz = move_z
         return dx,dy,dz
    
    for j in index_1:
        point = lattice_data.points[j]
        dx,dy,dz = f(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move_z)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz

def car_len(points,move):
    index_1 = np.where( 
        ((x_min + x_len* (9/31) <=  x) & (x <=  x_min + x_len* (21/31)))
     )[0]
     
    index_2 = np.where( 
        ( (x >  x_min + x_len* (21/31)))
     )[0]
     
     
    move = move / 5
    piv = x_min + x_len* (9/31)
    ll = x_len* (12/31)
    
    def f1(x,y,z,a):
         pc = (x-piv)/ll
         dx = pc*move
         dy = 0
         dz = 0
         return dx,dy,dz
    
    for j in index_1:
        point = lattice_data.points[j]
        dx,dy,dz = f1(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz
        
    def f2(x,y,z,a):
         dx = move
         dy = 0
         dz = 0
         return dx,dy,dz
    
    for j in index_2:
        point = lattice_data.points[j]
        dx,dy,dz = f2(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz


def car_width(points,move):
    y_mean = y.mean()
    index_1 = np.where( 
        ((y_mean - y_len* (1/7) <=  y) & (y <=  y_mean + y_len* (1/7)))
     )[0]
     
    index_2 = np.where( 
        ( y_mean - y_len* (1/7) >  y )
     )[0]
     
    index_3 = np.where( 
        ( y_mean + y_len* (1/7) <  y )
     )[0]
     
    move = move / 2.2
    move_half = move/2
    piv = y_mean
    ll =  y_len* (1/7) 
    
    def f1(x,y,z,a):
         pc = (y-piv)/ll
         dx = 0
         dy = pc*move_half
         dz = 0
         return dx,dy,dz
    
    for j in index_1:
        point = lattice_data.points[j]
        dx,dy,dz = f1(point.co_deform.x ,point.co_deform.y,point.co_deform.z,move)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz

    
    for j in index_2:
        point = lattice_data.points[j]
        point.co_deform.y += (-move_half)

    for j in index_3:
        point = lattice_data.points[j]
        point.co_deform.y += (move_half )


def car_size(points,fact):
   
    for point in lattice_data.points:
        point.co_deform.x *= fact
        point.co_deform.y *= fact
        point.co_deform.z *= fact

def car_front_hood_angle(points, angle_z):
    
    index_1 = np.where(
          (z >= z_min +  z_len* (4/7))  & (z <= z_min + z_len* (5/7))
        & (x <=  x_min + x_len* (9/31) ))[0]

    ramp_angle_index = [index_1]
    center = x_min + x_len* (9/31) 
    move_z =  np.tan(angle_z) *(x_len* (9/31))
    move_z = move_z*1.7
    def f(x,y,z,a):
        dx = 0
        dy = 0
        pc = (center - x)/(x_len* (9/31))
        dz = pc * move_z
        return dx,dy,dz
    
    for j in ramp_angle_index[0]:
        point = lattice_data.points[j]
        dx,dy,dz = f(point.co_deform.x ,point.co_deform.y,point.co_deform.z,angle_z)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz

def car_air_intake_angle(points, angle_x):
    
    index_1 = np.where(
          (z >= z_min +  z_len* (0/7))  & (z <= z_min + z_len* (3/7))
        & (x <=  x_min + x_len* (3/31) ))[0]

    ramp_angle_index = [index_1]
    center = z_min + z_len* (3/7) 
    move_x = np.tan(angle_x) *(z_len* (3/7) *1.7 )
    move_x = move_x/5
    def f(x,y,z,a):
        pc = (center - z)/(z_len* (3/7))
        dx = move_x*pc
        dy = 0
        dz = 0
        return dx,dy,dz
    
    for j in ramp_angle_index[0]:
        point = lattice_data.points[j]
        dx,dy,dz = f(point.co_deform.x ,point.co_deform.y,point.co_deform.z,angle_x)
        point.co_deform.x += dx
        point.co_deform.y += dy
        point.co_deform.z += dz



car_air_intake_angle(points, 3/180*3.1415926)
car_front_hood_angle(points,-3)
car_size(points,1.2)
car_width(points,0.1)
car_len(points,0.1) 
ramp_angle(points, 10/180*3.1415926) #-8  ~ 16
front_bumper_length(points, -0.10)
wind_screen_x( points,0.1) # +-0.05
wind_screen_z(points,0.1) # +-0.05
side_mirrors_x(points,0.1)
side_mirrors_z(points,0.1)
rear_window_x(points,0.05) 
rear_window_z(points,0.05) 
trunklid_angle(points, -10/180*3.1415926)
trunklid_x(points,0.05) 
trunklid_z(points,0.05) 
diffusor_angle(points, -10/180*3.1415926)
car_green_house_angle(points, 10/180*3.1415)


def FFD(params,names = ""):
    car_size(points,params[0])
    car_width(points,params[1])
    car_len(points,params[2])
    ramp_angle(points,params[3]/180*3.1415926)
    front_bumper_length(points,params[4])
    wind_screen_x(points,params[5])
    wind_screen_z(points,params[6])
    side_mirrors_x(points,params[7])
    side_mirrors_z(points,params[8])
    rear_window_x(points,params[9])
    rear_window_z(points,params[10])
    trunklid_angle(points,params[11]/180*3.1415926)
    trunklid_x(points,params[12])
    trunklid_z(points,params[13])
    diffusor_angle(points,params[14]/180*3.1415926)
    car_green_house_angle(points,params[15]/180*3.1415926)
    car_front_hood_angle(points,params[16]/180*3.1415926)    
    car_air_intake_angle(points,params[17]/180*3.1415926)


    for obj_name in imported_objects:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            
            dirr = export_dir+"/"+names
            if not os.path.exists(dirr):
                os.makedirs(dirr)

            export_path = os.path.join(dirr, f"{obj_name}.stl")
            
            bpy.ops.export_mesh.stl(
                filepath=export_path,
                use_selection=True,
                global_scale=1000,  
                use_mesh_modifiers=True
            )
            print(f"已导出: {export_path}")



reset_points()
def write_FFD(i, params):
    FFD(params,"%.5d"%i)
    bpy.context.view_layer.update()
    reset_points()

import sys

ids = int(sys.argv[4])
ide = int(sys.argv[5])
i = ids
for params in input_params[ids:ide]:
    write_FFD(i, params)
    i = i+1
    print(i,'/',ide)
