# trace generated using paraview version 5.12.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *
import sys
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'EnSight Reader'
kwWakeRefine0503case = EnSightReader(registrationName='KwWakeRefine0503.case', CaseFileName=sys.argv[1])

UpdatePipeline(time=0.0, proxy=kwWakeRefine0503case)

# create a new 'Extract Block'
extractBlock1 = ExtractBlock(registrationName='ExtractBlock1', Input=kwWakeRefine0503case)

# Properties modified on extractBlock1
extractBlock1.Selectors = ['/Root/Wheels_FrontSurface', '/Root/Wheels_RearSurface', '/Root/bodypart_01_Body_Open_EngineBayFlowSurface', '/Root/bodypart_02_UB_EngineBayFlowSurface', '/Root/bodypart_03_FastbackSurface', '/Root/bodypart_04_ExhaustSystem_EngineBayFlowSurface', '/Root/bodypart_07_MirrorSurface', '/Root/bodypart_08_EngineBayTrim_EngineBayFlowSurface', '/Root/bodypart_09_EngineAndGearbox_EngineBayFlowSurface', '/Root/bodypart_10_FrontGrills_EngineBayFlowSurface', '/Root/Porosity_EngineBayFlow']

UpdatePipeline(time=0.0, proxy=extractBlock1)

# create a new 'Programmable Filter'
programmableFilter1 = ProgrammableFilter(registrationName='ProgrammableFilter1', Input=extractBlock1)

# Properties modified on programmableFilter1
programmableFilter1.Script = """input_mb = self.GetInputDataObject(0, 0)
output_mb = self.GetOutput()

output_mb.ShallowCopy(input_mb)

for i in range(output_mb.GetNumberOfBlocks()):
    block = output_mb.GetBlock(i)
    if not block or not block.IsA("vtkDataSet"):
        continue
    
    cell_data = block.GetCellData()
    
    if not cell_data.GetArray("WallShearStress"):
        num_cells = block.GetNumberOfCells()
        
        wss_array = vtk.vtkFloatArray()
        wss_array.SetName("WallShearStress")
        wss_array.SetNumberOfComponents(3)
        wss_array.SetNumberOfTuples(num_cells)
        wss_array.Fill(0.0)   
        cell_data.AddArray(wss_array)
        
        for suffix in [\'i\', \'j\', \'k\', \'Magnitude\']:
            component_array = vtk.vtkFloatArray()
            component_array.SetName(f"WallShearStress{suffix}")
            component_array.SetNumberOfValues(num_cells)
            component_array.Fill(0.0)
            cell_data.AddArray(component_array)"""
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.PythonPath = ''

UpdatePipeline(time=0.0, proxy=programmableFilter1)

# create a new 'Cell Size'
cellSize1 = CellSize(registrationName='CellSize1', Input=programmableFilter1)

UpdatePipeline(time=0.0, proxy=cellSize1)

# create a new 'Merge Blocks'
mergeBlocks1 = MergeBlocks(registrationName='MergeBlocks1', Input=cellSize1)

UpdatePipeline(time=0.0, proxy=mergeBlocks1)

# create a new 'Extract Surface'
extractSurface1 = ExtractSurface(registrationName='ExtractSurface1', Input=mergeBlocks1)

UpdatePipeline(time=0.0, proxy=extractSurface1)

# set active source
SetActiveSource(programmableFilter1)

# set active source
SetActiveSource(cellSize1)

# set active source
SetActiveSource(mergeBlocks1)

# set active source
SetActiveSource(extractSurface1)

# create a new 'Generate Surface Normals'
generateSurfaceNormals1 = GenerateSurfaceNormals(registrationName='GenerateSurfaceNormals1', Input=extractSurface1)

# Properties modified on generateSurfaceNormals1
generateSurfaceNormals1.ComputeCellNormals = 1

UpdatePipeline(time=0.0, proxy=generateSurfaceNormals1)

# save data
SaveData(sys.argv[2], proxy=generateSurfaceNormals1, ChooseArraysToWrite=1,
    CellDataArrays=['Area', 'Normals', 'Pressure', 'WallShearStressi', 'WallShearStressj', 'WallShearStressk'])