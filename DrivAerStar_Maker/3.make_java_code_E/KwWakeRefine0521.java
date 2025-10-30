// Simcenter STAR-CCM+ macro: car0409.java
// Written by Simcenter STAR-CCM+ 18.06.007 
package macro;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.File;

import star.base.neo.*;
import star.turbulence.*;
import star.flow.*;
import star.metrics.*;
import star.meshing.*;
import star.common.*;
import star.material.*;
import star.keturb.*;
import star.base.report.*;
import star.coupledflow.*;
import star.prismmesher.*;
import star.vis.*;
import star.trimmer.*;
import star.common.*;
import star.segregatedflow.*;
import star.material.*;
import star.turbulence.*;
import star.flow.*;
import star.kwturb.*;
import star.metrics.*;

public class KwWakeRefine0521 extends StarMacro {

  public double floor = -285.1; // (mm)
  public double ARF = 2.0;
  public String back = "<<<back>>>";                               
  public double airDensity = 1.25;
  public double smallF = 1;
  public double car_length = 5 * smallF;
  
  public double[] Box1_xyz1; 
  public double[] Box1_xyz2;
  
  public double[] Box2_xyz1;     
  public double[] Box2_xyz2;
  
  public double[] Box3_xyz1;     
  public double[] Box3_xyz2;          

  public double[] Box4_xyz1;      
  public double[] Box4_xyz2;   
  

  public double inlet_velocity = 144.0;
  public double wallRelativeRotation = 0 ;
  public double global_mesh_size = smallF * 0.24;


  public void execute() {
    String dir = "<<<dir>>>/";
    floor = readfloor(dir);
    floor = floor/1000.0;
    floor = floor + 0.02;

    Box1_xyz1 = new double[]{-15.0 * smallF, -15.0 * smallF, floor * smallF};
    Box1_xyz2 = new double[]{45 * smallF, 15 * smallF, 15 * smallF};

    Box2_xyz1 = new double[]{-1.5 * smallF, -2 * smallF, floor * smallF};
    Box2_xyz2 = new double[]{9 * smallF, 2 * smallF, 3.1 * smallF};

    Box3_xyz1 = new double[]{-2.5 * smallF, -4 * smallF, floor * smallF};
    Box3_xyz2 = new double[]{3 * car_length, 4 * smallF, 4.8 * smallF};

    Box4_xyz1 = new double[]{3 * smallF, -1.4 * smallF, floor * smallF};
    Box4_xyz2 = new double[]{8.0 * smallF, 1.4 * smallF, 1.2 * smallF};


    wallRelativeRotation = - inlet_velocity/3.6/floor;
    execute_input(dir);
    execute0();
    executeMaximumNumberSteps();
    execute1(dir);
    execute2(dir);
    execute3();
    executeARF(dir);
    executeCD(dir);
    executekxl();
    setConstantDensityProperty();
    executeRun();
    execute2(dir);
    createDirectoryIfNotExists(dir+"case/");
    executeExportCase(dir+"case/");
  }


    public boolean createDirectoryIfNotExists(String directoryPath) {
        File directory = new File(directoryPath);
        if (!directory.exists()) {
            return directory.mkdirs();
        }
        return false;
    }


    public double readfloor(String dir) {
        String filePath = dir + "floor.txt";
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line = reader.readLine();
            if (line == null) {
                throw new RuntimeException("file error");
            }
            String filtered = line.replaceAll("[^-0-9.]", "");
            if (filtered.isEmpty()) {
                throw new RuntimeException("no info");
            }
            return Double.parseDouble(filtered);
        } catch (IOException e) {
            throw new RuntimeException("error: " + e.getMessage());
        } catch (NumberFormatException e) {
            throw new RuntimeException("NumberFormatException: " + e.getMessage());
        }
    }
   
  private void executekxl() {

    Simulation simulation_0 = 
      getActiveSimulation();

    Region region_1 = 
      simulation_0.getRegionManager().getRegion("Porosity_EngineBayFlow");

    VolumePorosityProfile volumePorosityProfile_0 = 
      region_1.getValues().get(VolumePorosityProfile.class);

    Units units_0 = 
      ((Units) simulation_0.getUnitsManager().getObject(""));

    volumePorosityProfile_0.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValueAndUnits(0.6, units_0);
  }

  private void executeRun() {
    Simulation simulation_0 = getActiveSimulation();
    simulation_0.getSimulationIterator().run(); //

  }


  private void executeExportCase(String dir) {

    Simulation simulation_0 = 
      getActiveSimulation();

    ImportManager importManager_0 = 
      simulation_0.getImportManager();

    importManager_0.setExportPath(dir+"KwWakeRefine0503.case");

    importManager_0.setFormatType(SolutionExportFormat.Type.CASE);

    importManager_0.setExportParts(new NeoObjectVector(new Object[] {}));

    importManager_0.setExportPartSurfaces(new NeoObjectVector(new Object[] {}));

    Region region_0 = 
      simulation_0.getRegionManager().getRegion("domain");

    Boundary boundary_1 = 
      region_0.getBoundaryManager().getBoundary("Block.bottom");

    Boundary boundary_3 = 
      region_0.getBoundaryManager().getBoundary("Block.inlet");

    Boundary boundary_4 = 
      region_0.getBoundaryManager().getBoundary("Block.outlet");

    Boundary boundary_2 = 
      region_0.getBoundaryManager().getBoundary("Block.side");

    Boundary boundary_0 = 
      region_0.getBoundaryManager().getBoundary("Block.top");

    Boundary boundary_5 = 
      region_0.getBoundaryManager().getBoundary("body.part_01_Body_Open_EngineBayFlow.Surface");

    Boundary boundary_6 = 
      region_0.getBoundaryManager().getBoundary("body.part_02_UB_EngineBayFlow.Surface");

    Boundary boundary_7 = 
      region_0.getBoundaryManager().getBoundary("body.part_03_Estate.Surface");

    Boundary boundary_8 = 
      region_0.getBoundaryManager().getBoundary("body.part_04_ExhaustSystem_EngineBayFlow.Surface");

    Boundary boundary_9 = 
      region_0.getBoundaryManager().getBoundary("body.part_07_Mirror.Surface");

    Boundary boundary_10 = 
      region_0.getBoundaryManager().getBoundary("body.part_08_EngineBayTrim_EngineBayFlow.Surface");

    Boundary boundary_11 = 
      region_0.getBoundaryManager().getBoundary("body.part_09_EngineAndGearbox_EngineBayFlow.Surface");

    Boundary boundary_12 = 
      region_0.getBoundaryManager().getBoundary("body.part_10_FrontGrills_EngineBayFlow.Surface");

    Boundary boundary_13 = 
      region_0.getBoundaryManager().getBoundary("Porosity_EngineBayFlow.Surface");

    InterfaceBoundary interfaceBoundary_0 = 
      ((InterfaceBoundary) region_0.getBoundaryManager().getBoundary("Porosity_EngineBayFlow.Surface [\u4EA4\u754C\u9762 1]"));

    Boundary boundary_14 = 
      region_0.getBoundaryManager().getBoundary("Wheels_Front.Surface");

    Boundary boundary_15 = 
      region_0.getBoundaryManager().getBoundary("Wheels_Rear.Surface");

    Region region_1 = 
      simulation_0.getRegionManager().getRegion("Porosity_EngineBayFlow");

    Boundary boundary_16 = 
      region_1.getBoundaryManager().getBoundary("Surface");

    InterfaceBoundary interfaceBoundary_1 = 
      ((InterfaceBoundary) region_1.getBoundaryManager().getBoundary("Surface [\u4EA4\u754C\u9762 1]"));

    importManager_0.setExportBoundaries(new NeoObjectVector(new Object[] {boundary_1, boundary_3, boundary_4, boundary_2, boundary_0, boundary_5, boundary_6, boundary_7, boundary_8, boundary_9, boundary_10, boundary_11, boundary_12, boundary_13, interfaceBoundary_0, boundary_14, boundary_15, boundary_16, interfaceBoundary_1}));

    importManager_0.setExportRegions(new NeoObjectVector(new Object[] {region_0, region_1}));

    PrimitiveFieldFunction primitiveFieldFunction_0 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("Pressure"));

    PrimitiveFieldFunction primitiveFieldFunction_1 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("Velocity"));

    VectorMagnitudeFieldFunction vectorMagnitudeFieldFunction_0 = 
      ((VectorMagnitudeFieldFunction) primitiveFieldFunction_1.getMagnitudeFunction());

    VectorComponentFieldFunction vectorComponentFieldFunction_0 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_1.getComponentFunction(0));

    VectorComponentFieldFunction vectorComponentFieldFunction_1 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_1.getComponentFunction(1));

    VectorComponentFieldFunction vectorComponentFieldFunction_2 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_1.getComponentFunction(2));

    PrimitiveFieldFunction primitiveFieldFunction_2 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("WallShearStress"));

    VectorMagnitudeFieldFunction vectorMagnitudeFieldFunction_1 = 
      ((VectorMagnitudeFieldFunction) primitiveFieldFunction_2.getMagnitudeFunction());

    VectorComponentFieldFunction vectorComponentFieldFunction_3 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_2.getComponentFunction(0));

    VectorComponentFieldFunction vectorComponentFieldFunction_4 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_2.getComponentFunction(1));

    VectorComponentFieldFunction vectorComponentFieldFunction_5 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_2.getComponentFunction(2));

    importManager_0.setExportScalars(new NeoObjectVector(new Object[] {primitiveFieldFunction_0, vectorMagnitudeFieldFunction_0, vectorComponentFieldFunction_0, vectorComponentFieldFunction_1, vectorComponentFieldFunction_2, vectorMagnitudeFieldFunction_1, vectorComponentFieldFunction_3, vectorComponentFieldFunction_4, vectorComponentFieldFunction_5}));

    importManager_0.setExportVectors(new NeoObjectVector(new Object[] {primitiveFieldFunction_1, primitiveFieldFunction_2}));

    importManager_0.setExportOptionAppendToFile(false);

    importManager_0.setExportOptionDataAtVerts(false);

    importManager_0.setExportOptionSolutionOnly(false);

    importManager_0.export(resolvePath(dir+"KwWakeRefine0503.case"), new NeoObjectVector(new Object[] {region_0, region_1}), new NeoObjectVector(new Object[] {boundary_1, boundary_3, boundary_4, boundary_2, boundary_0, boundary_5, boundary_6, boundary_7, boundary_8, boundary_9, boundary_10, boundary_11, boundary_12, boundary_13, interfaceBoundary_0, boundary_14, boundary_15, boundary_16, interfaceBoundary_1}), new NeoObjectVector(new Object[] {}), new NeoObjectVector(new Object[] {}), new NeoObjectVector(new Object[] {primitiveFieldFunction_0, vectorMagnitudeFieldFunction_0, vectorComponentFieldFunction_0, vectorComponentFieldFunction_1, vectorComponentFieldFunction_2, vectorMagnitudeFieldFunction_1, vectorComponentFieldFunction_3, vectorComponentFieldFunction_4, vectorComponentFieldFunction_5, primitiveFieldFunction_1, primitiveFieldFunction_2}), NeoProperty.fromString("{\'exportFormatType\': 2, \'appendToFile\': false, \'solutionOnly\': false, \'dataAtVerts\': false}"));
  }

  private void executeMaximumNumberSteps() {

    Simulation simulation_0 = getActiveSimulation();

    StepStoppingCriterion stepStoppingCriterion_0 = ((StepStoppingCriterion) simulation_0
        .getSolverStoppingCriterionManager().getSolverStoppingCriterion("Maximum Steps"));

    IntegerValue integerValue_0 = stepStoppingCriterion_0.getMaximumNumberStepsObject();

    integerValue_0.getQuantity().setValue(1000.0);
  }

  private void execute_input(String dir) {

    Simulation simulation_0 = getActiveSimulation();

    Units units_0 = simulation_0.getUnitsManager().getPreferredUnits(Dimensions.Builder().length(1).build());

    PartImportManager partImportManager_0 = simulation_0.get(PartImportManager.class);

    Units units_1 = ((Units) simulation_0.getUnitsManager().getObject("m"));

    partImportManager_0.importDbsParts(
        new StringVector(
            new String[] { resolvePath(dir + "Porosity_EngineBayFlow.dbs"), resolvePath(dir + "Wheels_Front.dbs"),
                resolvePath(dir + "Wheels_Rear.dbs"), resolvePath(dir + "body.dbs") }),
        "OneSurfacePerPatch", true, "OnePartPerFile", false, units_1, 1, false);

    simulation_0.getSceneManager().createGeometryScene("\u51E0\u4F55\u573A\u666F", "Outline", "Surface", 1, null);

    Scene scene_0 = simulation_0.getSceneManager().getScene("\u51E0\u4F55\u573A\u666F 1");

    scene_0.initializeAndWait();

    SceneUpdate sceneUpdate_0 = scene_0.getSceneUpdate();

    HardcopyProperties hardcopyProperties_0 = sceneUpdate_0.getHardcopyProperties();

    hardcopyProperties_0.setCurrentResolutionWidth(1648);

    hardcopyProperties_0.setCurrentResolutionHeight(548);

    scene_0.resetCamera();
  }

  private void execute0() {

    Simulation simulation_0 = getActiveSimulation();

    Scene scene_0 = simulation_0.getSceneManager().getScene("\u51E0\u4F55\u573A\u666F 1");

    CurrentView currentView_0 = scene_0.getCurrentView();

    Units units_0 = simulation_0.getUnitsManager().getPreferredUnits(Dimensions.Builder().length(1).build());

    scene_0.setTransparencyOverrideMode(SceneTransparencyOverride.MAKE_SCENE_TRANSPARENT);

    MeshPartFactory meshPartFactory_0 = simulation_0.get(MeshPartFactory.class);

    SimpleBlockPart simpleBlockPart_0 = meshPartFactory_0
        .createNewBlockPart(simulation_0.get(SimulationPartManager.class));

    simpleBlockPart_0.setDoNotRetessellate(true);

    LabCoordinateSystem labCoordinateSystem_0 = simulation_0.getCoordinateSystemManager().getLabCoordinateSystem();

    simpleBlockPart_0.setCoordinateSystem(labCoordinateSystem_0);

    simpleBlockPart_0.getCorner1().setCoordinateSystem(labCoordinateSystem_0);

    simpleBlockPart_0.getCorner1().setCoordinate(units_0, units_0, units_0,
        new DoubleVector(new double[] { -0.8107054233551025, -1.0149303674697876, -0.31850019097328186 }));

    simpleBlockPart_0.getCorner2().setCoordinateSystem(labCoordinateSystem_0);

    simpleBlockPart_0.getCorner2().setCoordinate(units_0, units_0, units_0,
        new DoubleVector(new double[] { 3.8049561977386475, 1.0149370431900024, 1.1032911539077759 }));

    simpleBlockPart_0.rebuildSimpleShapePart();

    scene_0.setTransparencyOverrideMode(SceneTransparencyOverride.USE_DISPLAYER_PROPERTY);

    simpleBlockPart_0.setDoNotRetessellate(false);

    simpleBlockPart_0.getCorner1().setCoordinate(units_0, units_0, units_0, new DoubleVector(Box1_xyz1));
    simpleBlockPart_0.getCorner2().setCoordinate(units_0, units_0, units_0, new DoubleVector(Box1_xyz2));


    SimpleBlockPart simpleBlockPart_1 = meshPartFactory_0
        .createNewBlockPart(simulation_0.get(SimulationPartManager.class));

    simpleBlockPart_1.setDoNotRetessellate(true);

    simpleBlockPart_1.setCoordinateSystem(labCoordinateSystem_0);

    simpleBlockPart_1.getCorner1().setCoordinateSystem(labCoordinateSystem_0);

    simpleBlockPart_1.getCorner2().setCoordinateSystem(labCoordinateSystem_0);

    simpleBlockPart_1.rebuildSimpleShapePart();

    simpleBlockPart_1.setDoNotRetessellate(false);

    simpleBlockPart_1.getCorner1().setCoordinate(units_0, units_0, units_0, new DoubleVector(Box2_xyz1));

    simpleBlockPart_1.getCorner2().setCoordinate(units_0, units_0, units_0, new DoubleVector(Box2_xyz2));

    // Block 3
    SimpleBlockPart simpleBlockPart_2 = meshPartFactory_0
        .createNewBlockPart(simulation_0.get(SimulationPartManager.class));

    simpleBlockPart_2.setDoNotRetessellate(true);

    simpleBlockPart_2.setCoordinateSystem(labCoordinateSystem_0);

    simpleBlockPart_2.getCorner1().setCoordinateSystem(labCoordinateSystem_0);

    simpleBlockPart_2.getCorner2().setCoordinateSystem(labCoordinateSystem_0);

    simpleBlockPart_2.rebuildSimpleShapePart();

    simpleBlockPart_2.setDoNotRetessellate(false);


    simpleBlockPart_2.getCorner1().setCoordinate(units_0, units_0, units_0, new DoubleVector(Box3_xyz1));

    simpleBlockPart_2.getCorner2().setCoordinate(units_0, units_0, units_0, new DoubleVector(Box3_xyz2));

    // Block 4
    SimpleBlockPart simpleBlockPart_3 = meshPartFactory_0
        .createNewBlockPart(simulation_0.get(SimulationPartManager.class));

    simpleBlockPart_3.setDoNotRetessellate(true);

    simpleBlockPart_3.setCoordinateSystem(labCoordinateSystem_0);

    simpleBlockPart_3.getCorner1().setCoordinateSystem(labCoordinateSystem_0);

    simpleBlockPart_3.getCorner2().setCoordinateSystem(labCoordinateSystem_0);

    simpleBlockPart_3.rebuildSimpleShapePart();

    simpleBlockPart_3.setDoNotRetessellate(false);

    simpleBlockPart_3.getCorner1().setCoordinate(units_0, units_0, units_0, new DoubleVector(Box4_xyz1));

    simpleBlockPart_3.getCorner2().setCoordinate(units_0, units_0, units_0, new DoubleVector(Box4_xyz2));

    // AutoMesh : Surface Control
    PartSurface partSurface_0 = ((PartSurface) simpleBlockPart_0.getPartSurfaceManager()
        .getPartSurface("Block Surface"));

    simpleBlockPart_0.splitPartSurfaceByPatch(partSurface_0, new IntVector(new int[] { 81 }), "inlet");

    simpleBlockPart_0.splitPartSurfaceByPatch(partSurface_0, new IntVector(new int[] { 84 }), "outlet");

    simpleBlockPart_0.splitPartSurfaceByPatch(partSurface_0, new IntVector(new int[] { 80, 82 }), "side");

    simpleBlockPart_0.splitPartSurfaceByPatch(partSurface_0, new IntVector(new int[] { 83 }), "bottom");

    partSurface_0.setPresentationName("top");

    MeshPart meshPart_0 = ((MeshPart) simulation_0.get(SimulationPartManager.class).getPart("body"));

    MeshPart meshPart_1 = ((MeshPart) simulation_0.get(SimulationPartManager.class).getPart("Porosity_EngineBayFlow"));

    MeshPart meshPart_2 = ((MeshPart) simulation_0.get(SimulationPartManager.class).getPart("Wheels_Front"));

    MeshPart meshPart_3 = ((MeshPart) simulation_0.get(SimulationPartManager.class).getPart("Wheels_Rear"));

    SubtractPartsOperation subtractPartsOperation_0 = (SubtractPartsOperation) simulation_0
        .get(MeshOperationManager.class).createSubtractPartsOperation(
            new NeoObjectVector(new Object[] { simpleBlockPart_0, meshPart_0, meshPart_1, meshPart_2, meshPart_3 }));

    subtractPartsOperation_0.getTargetPartManager().setQuery(null);

    subtractPartsOperation_0.getTargetPartManager().setObjects(simpleBlockPart_0);

    subtractPartsOperation_0.setPerformCADBoolean(true);

    subtractPartsOperation_0.execute();

    MeshOperationPart meshOperationPart_0 = ((MeshOperationPart) simulation_0.get(SimulationPartManager.class)
        .getPart("\u51CF\u8FD0\u7B97"));

    meshOperationPart_0.setPresentationName("domain");

    simulation_0.getRegionManager().newRegionsFromParts(
        new NeoObjectVector(new Object[] { meshOperationPart_0, meshPart_1 }), "OneRegionPerPart", null,
        "OneBoundaryPerPartSurface", null, "OneFeatureCurve", null, RegionManager.CreateInterfaceMode.BOUNDARY,
        "OneEdgeBoundaryPerPart", null);

    Region region_0 = simulation_0.getRegionManager().getRegion("Porosity_EngineBayFlow");

    PorousRegion porousRegion_0 = ((PorousRegion) simulation_0.get(ConditionTypeManager.class).get(PorousRegion.class));

    region_0.setRegionType(porousRegion_0);

    Region region_1 = simulation_0.getRegionManager().getRegion("domain");

    Boundary boundary_0 = region_1.getBoundaryManager().getBoundary("Porosity_EngineBayFlow.Surface");

    Boundary boundary_1 = region_0.getBoundaryManager().getBoundary("Surface");

    BoundaryInterface boundaryInterface_0 = simulation_0.getInterfaceManager().createBoundaryInterface(boundary_0,
        boundary_1, "\u4EA4\u754C\u9762");

    AutoMeshOperation autoMeshOperation_0 = simulation_0.get(MeshOperationManager.class)
        .createAutoMeshOperation(new StringVector(new String[] { "star.resurfacer.ResurfacerAutoMesher",
            "star.trimmer.TrimmerAutoMesher", "star.prismmesher.PrismAutoMesher" }),
            new NeoObjectVector(new Object[] { meshOperationPart_0 }));

    autoMeshOperation_0.getMesherParallelModeOption().setSelected(MesherParallelModeOption.Type.PARALLEL);

    // set global meshsize
    autoMeshOperation_0.getDefaultValues().get(BaseSize.class).setValueAndUnits(global_mesh_size, units_0);  

    NumPrismLayers numPrismLayers_0 = autoMeshOperation_0.getDefaultValues().get(NumPrismLayers.class);

    IntegerValue integerValue_0 = numPrismLayers_0.getNumLayersValue();

    integerValue_0.getQuantity().setValue(6.0);

    PrismLayerStretching prismLayerStretching_0 = autoMeshOperation_0.getDefaultValues()
        .get(PrismLayerStretching.class);

    Units units_1 = ((Units) simulation_0.getUnitsManager().getObject(""));

    prismLayerStretching_0.getStretchingQuantity().setValueAndUnits(1.2, units_1);

    PrismThickness prismThickness_0 = autoMeshOperation_0.getDefaultValues().get(PrismThickness.class);

    prismThickness_0.getRelativeSizeScalar().setValueAndUnits(10.0, units_1);

    PartsSimpleTemplateGrowthRate partsSimpleTemplateGrowthRate_0 = autoMeshOperation_0.getDefaultValues()
        .get(PartsSimpleTemplateGrowthRate.class);

    partsSimpleTemplateGrowthRate_0.getGrowthRateOption().setSelected(PartsGrowthRateOption.Type.VERYSLOW);

    MaximumCellSize maximumCellSize_0 = autoMeshOperation_0.getDefaultValues().get(MaximumCellSize.class);

    maximumCellSize_0.getRelativeSizeScalar().setValueAndUnits(200.0, units_1);

    SurfaceCustomMeshControl surfaceCustomMeshControl_0 = autoMeshOperation_0.getCustomMeshControls()
        .createSurfaceControl();

    GeometryObjectProxy geometryObjectProxy_0 = meshOperationPart_0.getOrCreateProxyForObject(meshPart_0);

    GeometryObjectProxy geometryObjectProxy_1 = meshOperationPart_0.getOrCreateProxyForObject(meshPart_1);

    GeometryObjectProxy geometryObjectProxy_2 = meshOperationPart_0.getOrCreateProxyForObject(meshPart_2);

    GeometryObjectProxy geometryObjectProxy_3 = meshOperationPart_0.getOrCreateProxyForObject(meshPart_3);

    surfaceCustomMeshControl_0.getGeometryObjects().setQuery(null);

    surfaceCustomMeshControl_0.getGeometryObjects().setObjects(geometryObjectProxy_0, geometryObjectProxy_1,
        geometryObjectProxy_2, geometryObjectProxy_3);

    surfaceCustomMeshControl_0.getCustomConditions().get(PartsMinimumSurfaceSizeOption.class)
        .setSelected(PartsMinimumSurfaceSizeOption.Type.CUSTOM);

    surfaceCustomMeshControl_0.getCustomConditions().get(PartsTargetSurfaceSizeOption.class)
        .setSelected(PartsTargetSurfaceSizeOption.Type.CUSTOM);

    PartsTargetSurfaceSize partsTargetSurfaceSize_0 = surfaceCustomMeshControl_0.getCustomValues()
        .get(PartsTargetSurfaceSize.class);

    partsTargetSurfaceSize_0.getRelativeSizeScalar().setValueAndUnits(6.0, units_1);

    PartsMinimumSurfaceSize partsMinimumSurfaceSize_0 = surfaceCustomMeshControl_0.getCustomValues()
        .get(PartsMinimumSurfaceSize.class);

    partsMinimumSurfaceSize_0.getRelativeSizeScalar().setValueAndUnits(4.0, units_1);

    SurfaceCustomMeshControl surfaceCustomMeshControl_1 = autoMeshOperation_0.getCustomMeshControls()
        .createSurfaceControl();

    surfaceCustomMeshControl_1.getGeometryObjects().setQuery(null);

    MeshOperationPart meshOperationPart_1 = 
      ((MeshOperationPart) simulation_0.get(SimulationPartManager.class).getPart("domain"));

    PartSurface partSurface_4 = 
      ((PartSurface) meshOperationPart_1.getPartSurfaceManager().getPartSurface("Block.inlet"));

    PartSurface partSurface_1 = 
      ((PartSurface) meshOperationPart_1.getPartSurfaceManager().getPartSurface("Block.outlet"));

    PartSurface partSurface_2 = 
      ((PartSurface) meshOperationPart_1.getPartSurfaceManager().getPartSurface("Block.side"));

    PartSurface partSurface_3 = 
      ((PartSurface) meshOperationPart_1.getPartSurfaceManager().getPartSurface("Block.top"));

      surfaceCustomMeshControl_1.getGeometryObjects().setObjects(partSurface_4, partSurface_1, partSurface_2, partSurface_3);
  

    PartsCustomizePrismMesh partsCustomizePrismMesh_0 = surfaceCustomMeshControl_1.getCustomConditions()
        .get(PartsCustomizePrismMesh.class);

    partsCustomizePrismMesh_0.getCustomPrismOptions().setSelected(PartsCustomPrismsOption.Type.DISABLE);

    surfaceCustomMeshControl_1.getCustomConditions().get(PartsMinimumSurfaceSizeOption.class)
        .setSelected(PartsMinimumSurfaceSizeOption.Type.CUSTOM);

    surfaceCustomMeshControl_1.getCustomConditions().get(PartsTargetSurfaceSizeOption.class)
        .setSelected(PartsTargetSurfaceSizeOption.Type.CUSTOM);

    PartsTargetSurfaceSize partsTargetSurfaceSize_1 = surfaceCustomMeshControl_1.getCustomValues()
        .get(PartsTargetSurfaceSize.class);

    partsTargetSurfaceSize_1.getRelativeSizeScalar().setValueAndUnits(200.0, units_1);

    PartsMinimumSurfaceSize partsMinimumSurfaceSize_1 = surfaceCustomMeshControl_1.getCustomValues()
        .get(PartsMinimumSurfaceSize.class);

    partsMinimumSurfaceSize_1.getRelativeSizeScalar().setValueAndUnits(200.0, units_1);

    // AutoMesh : Volume control - Block  
    VolumeCustomMeshControl volumeCustomMeshControl_0 = autoMeshOperation_0.getCustomMeshControls()
        .createVolumeControl();

    volumeCustomMeshControl_0.getGeometryObjects().setQuery(null);

    volumeCustomMeshControl_0.getGeometryObjects().setObjects(simpleBlockPart_1);

    VolumeControlTrimmerSizeOption volumeControlTrimmerSizeOption_0 = volumeCustomMeshControl_0.getCustomConditions()
        .get(VolumeControlTrimmerSizeOption.class);

    volumeControlTrimmerSizeOption_0.setVolumeControlBaseSizeOption(true);

    VolumeControlSize volumeControlSize_0 = volumeCustomMeshControl_0.getCustomValues().get(VolumeControlSize.class);

    volumeControlSize_0.getRelativeSizeScalar().setValueAndUnits(25.0, units_1);

    // AutoMesh : Volume control - Block 2
    VolumeCustomMeshControl volumeCustomMeshControl_1 = autoMeshOperation_0.getCustomMeshControls()
        .createVolumeControl();

    volumeCustomMeshControl_1.getGeometryObjects().setQuery(null);

    volumeCustomMeshControl_1.getGeometryObjects().setObjects(simpleBlockPart_2);

    VolumeControlTrimmerSizeOption volumeControlTrimmerSizeOption_1 = volumeCustomMeshControl_1.getCustomConditions()
        .get(VolumeControlTrimmerSizeOption.class);

    volumeControlTrimmerSizeOption_1.setVolumeControlBaseSizeOption(true);

    VolumeControlSize volumeControlSize_1 = volumeCustomMeshControl_1.getCustomValues().get(VolumeControlSize.class);

    volumeControlSize_1.getRelativeSizeScalar().setValueAndUnits(100.0, units_1);

    AutoMeshOperation autoMeshOperation_1 = simulation_0.get(MeshOperationManager.class).createAutoMeshOperation(
        new StringVector(new String[] { "star.resurfacer.ResurfacerAutoMesher", "star.trimmer.TrimmerAutoMesher" }),
        new NeoObjectVector(new Object[] { meshPart_1 }));

    autoMeshOperation_1.getMesherParallelModeOption().setSelected(MesherParallelModeOption.Type.PARALLEL);

    autoMeshOperation_1.getDefaultValues().get(BaseSize.class).setValueAndUnits(0.08*smallF, units_0);

    PartsSimpleTemplateGrowthRate partsSimpleTemplateGrowthRate_1 = autoMeshOperation_0.getDefaultValues()
        .get(PartsSimpleTemplateGrowthRate.class);

    partsSimpleTemplateGrowthRate_1.getGrowthRateOption().setSelected(PartsGrowthRateOption.Type.VERYSLOW);

    MaximumCellSize maximumCellSize_1 = autoMeshOperation_1.getDefaultValues().get(MaximumCellSize.class);

    maximumCellSize_1.getRelativeSizeScalar().setValueAndUnits(50.0, units_1);

    //  AutoMesh : Volume control - Block 3
    VolumeCustomMeshControl volumeCustomMeshControl_2 = autoMeshOperation_1.getCustomMeshControls()
        .createVolumeControl();

    volumeCustomMeshControl_2.getGeometryObjects().setQuery(null);

    volumeCustomMeshControl_2.getGeometryObjects().setObjects(meshPart_1);

    VolumeControlTrimmerSizeOption volumeControlTrimmerSizeOption_2 = volumeCustomMeshControl_2.getCustomConditions()
        .get(VolumeControlTrimmerSizeOption.class);

    volumeControlTrimmerSizeOption_2.setVolumeControlBaseSizeOption(true);

    VolumeControlSize volumeControlSize_2 = volumeCustomMeshControl_2.getCustomValues().get(VolumeControlSize.class);

    // 25% percent of global mesh size
    volumeControlSize_2.getRelativeSizeScalar().setValueAndUnits(25.0, units_1);

    // AutoMesh : Volume control - Block 4
    VolumeCustomMeshControl volumeCustomMeshControl_3 = autoMeshOperation_0.getCustomMeshControls()
        .createVolumeControl();

    volumeCustomMeshControl_3.getGeometryObjects().setQuery(null);

    volumeCustomMeshControl_3.getGeometryObjects().setObjects(simpleBlockPart_3);

    VolumeControlTrimmerSizeOption volumeControlTrimmerSizeOption_3 = volumeCustomMeshControl_3.getCustomConditions()
        .get(VolumeControlTrimmerSizeOption.class);

    volumeControlTrimmerSizeOption_3.setVolumeControlBaseSizeOption(true);

    VolumeControlSize volumeControlSize_3 = volumeCustomMeshControl_3.getCustomValues().get(VolumeControlSize.class);

    // 10% percent of global mesh size
    volumeControlSize_3.getRelativeSizeScalar().setValueAndUnits(10.0, units_1);

    // Physics
    PhysicsContinuum physicsContinuum_0 = simulation_0.getContinuumManager().createContinuum(PhysicsContinuum.class);

     physicsContinuum_0.enable(ThreeDimensionalModel.class);

    physicsContinuum_0.enable(SingleComponentGasModel.class);

    physicsContinuum_0.enable(SegregatedFlowModel.class);

    physicsContinuum_0.enable(ConstantDensityModel.class);

    physicsContinuum_0.enable(SteadyModel.class);

    physicsContinuum_0.enable(TurbulentModel.class);

    physicsContinuum_0.enable(RansTurbulenceModel.class);

    physicsContinuum_0.enable(KOmegaTurbulence.class);

    physicsContinuum_0.enable(SstKwTurbModel.class);

    physicsContinuum_0.enable(KwAllYplusWallTreatment.class);

    VelocityProfile velocityProfile_0 = physicsContinuum_0.getInitialConditions().get(VelocityProfile.class);

    Units units_2 = ((Units) simulation_0.getUnitsManager().getObject("kph"));

    velocityProfile_0.getMethod(ConstantVectorProfileMethod.class).getQuantity().setComponentsAndUnits(inlet_velocity,
        0.0, 0.0, units_2);

    Boundary boundary_2 = region_1.getBoundaryManager().getBoundary("Block.inlet");

    InletBoundary inletBoundary_0 = ((InletBoundary) simulation_0.get(ConditionTypeManager.class)
        .get(InletBoundary.class));

    boundary_2.setBoundaryType(inletBoundary_0);

    Boundary boundary_3 = region_1.getBoundaryManager().getBoundary("Block.outlet");

    PressureBoundary pressureBoundary_0 = ((PressureBoundary) simulation_0.get(ConditionTypeManager.class)
        .get(PressureBoundary.class));

    boundary_3.setBoundaryType(pressureBoundary_0);

    Boundary boundary_4 = region_1.getBoundaryManager().getBoundary("Block.side");

    SymmetryBoundary symmetryBoundary_0 = ((SymmetryBoundary) simulation_0.get(ConditionTypeManager.class)
        .get(SymmetryBoundary.class));

    boundary_4.setBoundaryType(symmetryBoundary_0);

    Boundary boundary_5 = region_1.getBoundaryManager().getBoundary("Block.top");

    boundary_5.setBoundaryType(symmetryBoundary_0);



      ///--------------------------- MG
         
      Boundary boundary_bottom = 
      region_1.getBoundaryManager().getBoundary("Block.bottom");

    boundary_bottom.getConditions().get(WallSlidingOption.class).setSelected(WallSlidingOption.Type.VECTOR);

    WallRelativeVelocityProfile wallRelativeVelocityProfile_bottom = 
    boundary_bottom.getValues().get(WallRelativeVelocityProfile.class);

    Units units_kph = 
      ((Units) simulation_0.getUnitsManager().getObject("kph"));

      wallRelativeVelocityProfile_bottom.getMethod(ConstantVectorProfileMethod.class).getQuantity().setComponentsAndUnits( inlet_velocity, 0.0, 0.0, units_kph);

         
        ///---------------------------
        /// 
        /// 
    VelocityMagnitudeProfile velocityMagnitudeProfile_0 = boundary_2.getValues().get(VelocityMagnitudeProfile.class);

    Units units_3 = ((Units) simulation_0.getUnitsManager().getObject("m/s"));

    velocityMagnitudeProfile_0.getMethod(ConstantScalarProfileMethod.class).getQuantity()
        .setValueAndUnits(inlet_velocity, units_3);

    velocityMagnitudeProfile_0.getMethod(ConstantScalarProfileMethod.class).getQuantity()
        .setValueAndUnits(inlet_velocity, units_2);

    PorousInertialResistance porousInertialResistance_0 = region_0.getValues().get(PorousInertialResistance.class);

    porousInertialResistance_0.setMethod(IsotropicTensorProfileMethod.class);

    PorousViscousResistance porousViscousResistance_0 = region_0.getValues().get(PorousViscousResistance.class);

    porousViscousResistance_0.setMethod(IsotropicTensorProfileMethod.class);

    ScalarProfile scalarProfile_0 = porousInertialResistance_0.getMethod(IsotropicTensorProfileMethod.class)
        .getIsotropicProfile();

    Units units_4 = ((Units) simulation_0.getUnitsManager().getObject("kg/m^4"));

    scalarProfile_0.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValueAndUnits(50.0, units_4);

    ScalarProfile scalarProfile_1 = porousViscousResistance_0.getMethod(IsotropicTensorProfileMethod.class)
        .getIsotropicProfile();

    Units units_5 = ((Units) simulation_0.getUnitsManager().getObject("kg/m^3-s"));

    scalarProfile_1.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValueAndUnits(2500.0, units_5);

    CartesianCoordinateSystem cartesianCoordinateSystem_0 = labCoordinateSystem_0.getLocalCoordinateSystemManager()
        .createLocalCoordinateSystem(CartesianCoordinateSystem.class, "\u7B1B\u5361\u5C14");

    cartesianCoordinateSystem_0.getOrigin().setCoordinate(units_0, units_0, units_0,
        new DoubleVector(new double[] { 0.0, 0.0, 0.0 }));

    cartesianCoordinateSystem_0.getOrigin().setUnits0(units_0);

    cartesianCoordinateSystem_0.getOrigin().setUnits1(units_0);

    cartesianCoordinateSystem_0.getOrigin().setUnits2(units_0);

    cartesianCoordinateSystem_0.getOrigin().setDefinition("");

    cartesianCoordinateSystem_0.getOrigin().setValue(new DoubleVector(new double[] { 0.0, 0.0, 0.0 }));

    cartesianCoordinateSystem_0.getXVector().setComponents(1.0, 0.0, 0.0);

    cartesianCoordinateSystem_0.getXyPlane().setComponents(0.0, 1.0, 0.0);

    CartesianCoordinateSystem cartesianCoordinateSystem_1 = labCoordinateSystem_0.getLocalCoordinateSystemManager()
        .createLocalCoordinateSystem(CartesianCoordinateSystem.class, "\u7B1B\u5361\u5C14");

    cartesianCoordinateSystem_1.getOrigin().setCoordinate(units_0, units_0, units_0,
        new DoubleVector(new double[] { 0.0, 0.0, 0.0 }));

    cartesianCoordinateSystem_1.getOrigin().setUnits0(units_0);

    cartesianCoordinateSystem_1.getOrigin().setUnits1(units_0);

    cartesianCoordinateSystem_1.getOrigin().setUnits2(units_0);

    cartesianCoordinateSystem_1.getOrigin().setDefinition("");

    cartesianCoordinateSystem_1.getOrigin().setValue(new DoubleVector(new double[] { 0.0, 0.0, 0.0 }));

    cartesianCoordinateSystem_1.getXVector().setComponents(1.0, 0.0, 0.0);

    cartesianCoordinateSystem_1.getXyPlane().setComponents(0.0, 1.0, 0.0);

    cartesianCoordinateSystem_0.setPresentationName("front");

    cartesianCoordinateSystem_1.setPresentationName("rear");

    cartesianCoordinateSystem_1.getOrigin().setCoordinate(units_0, units_0, units_0,
        new DoubleVector(new double[] { 2.8, 0.0, 0.0 }));

    Boundary boundary_6 = region_1.getBoundaryManager().getBoundary("Wheels_Front.Surface");

    boundary_6.getConditions().get(WallSlidingOption.class).setSelected(WallSlidingOption.Type.LOCAL_ROTATION_RATE);

    simulation_0.println("print Infomation : >>>>>> Line 614 ckpt 1 !");

    LocalAxis localAxis_0 = boundary_6.getValues().get(LocalAxis.class);

    LocalAxisLeaf localAxisLeaf_0 = localAxis_0.getModelPartValue();

    Units units_10 = ((Units) simulation_0.getUnitsManager().getObject(""));

    localAxisLeaf_0.getAxisVector().setComponentsAndUnits(0.0, -1.0, 0.0, units_10);

    LabCoordinateSystem labCoordinateSystem_00 = simulation_0.getCoordinateSystemManager().getLabCoordinateSystem();

    CartesianCoordinateSystem cartesianCoordinateSystem_00 = ((CartesianCoordinateSystem) labCoordinateSystem_00
        .getLocalCoordinateSystemManager().getObject("front"));

    localAxisLeaf_0.setCoordinateSystem(cartesianCoordinateSystem_00);


    WallRelativeRotationProfile wallRelativeRotationProfile_0 = boundary_6.getValues()
        .get(WallRelativeRotationProfile.class);

    Units units_6 = ((Units) simulation_0.getUnitsManager().getObject("radian/s"));

    wallRelativeRotationProfile_0.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValueAndUnits(wallRelativeRotation,
        units_6);

    
        //-----------------------

    Boundary boundary_8 = region_1.getBoundaryManager().getBoundary("Wheels_Rear.Surface");

    boundary_8.copyProperties(boundary_6);


    // ReferenceFrame referenceFrame_1 =
    // boundary_8.getValues().get(ReferenceFrame.class);

    LocalAxis localAxis_01 = boundary_8.getValues().get(LocalAxis.class);

    LocalAxisLeaf localAxisLeaf_01 = localAxis_01.getModelPartValue();

    localAxisLeaf_01.getAxisVector().setComponentsAndUnits(0.0, -1.0, 0.0, units_10);

    LabCoordinateSystem labCoordinateSystem_01 = simulation_0.getCoordinateSystemManager().getLabCoordinateSystem();

    CartesianCoordinateSystem cartesianCoordinateSystem_01 = ((CartesianCoordinateSystem) labCoordinateSystem_01
        .getLocalCoordinateSystemManager().getObject("rear"));

    localAxisLeaf_01.setCoordinateSystem(cartesianCoordinateSystem_01);

    LocalAxis localAxis_02 = boundary_8.getValues().get(LocalAxis.class);

    LocalAxisLeaf localAxisLeaf_02 = localAxis_02.getModelPartValue();

    localAxisLeaf_02.getAxisVector().setComponentsAndUnits(0.0, -1.0, 0.0, units_10);

    LabCoordinateSystem labCoordinateSystem_02 = simulation_0.getCoordinateSystemManager().getLabCoordinateSystem();

    CartesianCoordinateSystem cartesianCoordinateSystem_02 = ((CartesianCoordinateSystem) labCoordinateSystem_02
        .getLocalCoordinateSystemManager().getObject("rear"));

    localAxisLeaf_02.setCoordinateSystem(cartesianCoordinateSystem_02);


  }

  private void execute1(String dir) {

    Simulation simulation_0 = getActiveSimulation();

    Units units_0 = simulation_0.getUnitsManager().getPreferredUnits(Dimensions.Builder().length(1).build());

    Scene scene_0 = simulation_0.getSceneManager().getScene("\u51E0\u4F55\u573A\u666F 1");

    scene_0.setTransparencyOverrideMode(SceneTransparencyOverride.MAKE_SCENE_TRANSPARENT);

    scene_0.getCreatorGroup().setQuery(null);

    Region region_1 = simulation_0.getRegionManager().getRegion("domain");

    Region region_0 = simulation_0.getRegionManager().getRegion("Porosity_EngineBayFlow");

    scene_0.getCreatorGroup().setObjects(region_1, region_0);

    scene_0.getCreatorGroup().setQuery(null);

    scene_0.getCreatorGroup().setObjects(region_1, region_0);

    PlaneSection planeSection_0 = (PlaneSection) simulation_0.getPartManager().createImplicitPart(
        new NeoObjectVector(new Object[] {}), new DoubleVector(new double[] { 0.0, 0.0, 1.0 }),
        new DoubleVector(new double[] { 0.0, 0.0, 0.0 }), 0, 1, new DoubleVector(new double[] { 0.0 }));

    LabCoordinateSystem labCoordinateSystem_0 = simulation_0.getCoordinateSystemManager().getLabCoordinateSystem();

    planeSection_0.setCoordinateSystem(labCoordinateSystem_0);

    planeSection_0.getInputParts().setQuery(null);

    planeSection_0.getInputParts().setObjects(region_1, region_0);

    scene_0.setTransparencyOverrideMode(SceneTransparencyOverride.USE_DISPLAYER_PROPERTY);

    planeSection_0.getOriginCoordinate().setUnits0(units_0);

    planeSection_0.getOriginCoordinate().setUnits1(units_0);

    planeSection_0.getOriginCoordinate().setUnits2(units_0);

    planeSection_0.getOriginCoordinate().setDefinition("");

    planeSection_0.getOriginCoordinate()
        .setValue(new DoubleVector(new double[] { 1.4971253871917725, 3.337860107421875E-6, 0.392395481467247 }));

    planeSection_0.getOriginCoordinate().setCoordinate(units_0, units_0, units_0,
        new DoubleVector(new double[] { 1.4971253871917725, 3.337860107421875E-6, 0.392395481467247 }));

    planeSection_0.getOriginCoordinate().setCoordinateSystem(labCoordinateSystem_0);

    planeSection_0.getOrientationCoordinate().setUnits0(units_0);

    planeSection_0.getOrientationCoordinate().setUnits1(units_0);

    planeSection_0.getOrientationCoordinate().setUnits2(units_0);

    planeSection_0.getOrientationCoordinate().setDefinition("");

    planeSection_0.getOrientationCoordinate().setValue(new DoubleVector(new double[] { 0.0, 1.0, 0.0 }));

    planeSection_0.getOrientationCoordinate().setCoordinate(units_0, units_0, units_0,
        new DoubleVector(new double[] { 0.0, 1.0, 0.0 }));

    planeSection_0.getOrientationCoordinate().setCoordinateSystem(labCoordinateSystem_0);

    SingleValue singleValue_0 = planeSection_0.getSingleValue();

    singleValue_0.getValueQuantity().setValue(0.0);

    singleValue_0.getValueQuantity().setUnits(units_0);

    RangeMultiValue rangeMultiValue_0 = planeSection_0.getRangeMultiValue();

    rangeMultiValue_0.setNValues(2);

    rangeMultiValue_0.getStartQuantity().setValue(0.0);

    rangeMultiValue_0.getStartQuantity().setUnits(units_0);

    rangeMultiValue_0.getEndQuantity().setValue(1.0);

    rangeMultiValue_0.getEndQuantity().setUnits(units_0);

    DeltaMultiValue deltaMultiValue_0 = planeSection_0.getDeltaMultiValue();

    deltaMultiValue_0.setNValues(2);

    deltaMultiValue_0.getStartQuantity().setValue(0.0);

    deltaMultiValue_0.getStartQuantity().setUnits(units_0);

    deltaMultiValue_0.getDeltaQuantity().setValue(1.0);

    deltaMultiValue_0.getDeltaQuantity().setUnits(units_0);

    MultiValue multiValue_0 = planeSection_0.getArbitraryMultiValue();

    multiValue_0.getValueQuantities().setUnits(units_0);

    multiValue_0.getValueQuantities().setArray(new DoubleVector(new double[] { 0.0 }));

    planeSection_0.setValueMode(ValueMode.SINGLE);

    ForceReport forceReport_0 = simulation_0.getReportManager().createReport(ForceReport.class);

    forceReport_0.setPresentationName("drag");

    forceReport_0.getParts().setQuery(null);

    Boundary boundary_2 = region_1.getBoundaryManager().getBoundary("Block.inlet");

    Boundary boundary_10 = region_1.getBoundaryManager().getBoundary("body.part_01_Body_Open_EngineBayFlow.Surface");

    Boundary boundary_14 = region_1.getBoundaryManager().getBoundary("body.part_02_UB_EngineBayFlow.Surface");

    Boundary boundary_15 = region_1.getBoundaryManager().getBoundary("body.part_03_"+back+".Surface");


    Boundary boundary_17 = region_1.getBoundaryManager()
        .getBoundary("body.part_04_ExhaustSystem_EngineBayFlow.Surface");

    Boundary boundary_18 = region_1.getBoundaryManager().getBoundary("body.part_07_Mirror.Surface");


    Boundary boundary_20 = region_1.getBoundaryManager()
        .getBoundary("body.part_08_EngineBayTrim_EngineBayFlow.Surface");

    Boundary boundary_21 = region_1.getBoundaryManager()
        .getBoundary("body.part_09_EngineAndGearbox_EngineBayFlow.Surface");

    Boundary boundary_22 = region_1.getBoundaryManager()
        .getBoundary("body.part_10_FrontGrills_EngineBayFlow.Surface");

    Boundary boundary_0 = region_1.getBoundaryManager().getBoundary("Porosity_EngineBayFlow.Surface");

    InterfaceBoundary interfaceBoundary_0 = ((InterfaceBoundary) region_1.getBoundaryManager()
        .getBoundary("Porosity_EngineBayFlow.Surface [\u4EA4\u754C\u9762 1]"));

    Boundary boundary_6 = region_1.getBoundaryManager().getBoundary("Wheels_Front.Surface");


    Boundary boundary_8 = region_1.getBoundaryManager().getBoundary("Wheels_Rear.Surface");


    forceReport_0.getParts().setObjects(boundary_2, boundary_10,  boundary_14,
        boundary_15, boundary_17, boundary_18, boundary_20, boundary_21, boundary_22,
        boundary_0, interfaceBoundary_0, boundary_6, boundary_8);

    simulation_0.getMonitorManager().createMonitorAndPlot(new NeoObjectVector(new Object[] { forceReport_0 }), true,
        "%1$s \u7ED8\u56FE");

    ReportMonitor reportMonitor_0 = ((ReportMonitor) simulation_0.getMonitorManager().getMonitor("drag Monitor"));

    MonitorPlot monitorPlot_0 = simulation_0.getPlotManager()
        .createMonitorPlot(new NeoObjectVector(new Object[] { reportMonitor_0 }), "drag Monitor \u7ED8\u56FE");

    monitorPlot_0.openInteractive();

    PlotUpdate plotUpdate_0 = monitorPlot_0.getPlotUpdate();

    HardcopyProperties hardcopyProperties_1 = plotUpdate_0.getHardcopyProperties();

    hardcopyProperties_1.setCurrentResolutionWidth(25);

    hardcopyProperties_1.setCurrentResolutionHeight(25);

    SceneUpdate sceneUpdate_0 = scene_0.getSceneUpdate();

    HardcopyProperties hardcopyProperties_0 = sceneUpdate_0.getHardcopyProperties();

    hardcopyProperties_0.setCurrentResolutionWidth(1033);

    hardcopyProperties_0.setCurrentResolutionHeight(483);

    hardcopyProperties_1.setCurrentResolutionWidth(1031);

    hardcopyProperties_1.setCurrentResolutionHeight(482);

    hardcopyProperties_0.setCurrentResolutionWidth(1031);

    hardcopyProperties_0.setCurrentResolutionHeight(482);

    simulation_0.saveState(dir + "star.KwWakeRefine0521.sim");
  }

  private void execute2(String dir) {

    Simulation simulation_0 = getActiveSimulation();

    MeshPipelineController meshPipelineController_0 = simulation_0.get(MeshPipelineController.class);

    meshPipelineController_0.generateVolumeMesh();

    simulation_0.saveState(dir + "star.KwWakeRefine0521.sim");
  }

  private void execute3() {

    Simulation simulation_0 = getActiveSimulation();

    simulation_0.getSceneManager().createGeometryScene("\u7F51\u683C\u573A\u666F", "\u8F6E\u5ED3", "\u7F51\u683C", 3);

    Scene scene_1 = simulation_0.getSceneManager().getScene("\u7F51\u683C\u573A\u666F 1");

    scene_1.initializeAndWait();

    SceneUpdate sceneUpdate_1 = scene_1.getSceneUpdate();

    HardcopyProperties hardcopyProperties_2 = sceneUpdate_1.getHardcopyProperties();

    hardcopyProperties_2.setCurrentResolutionWidth(25);

    hardcopyProperties_2.setCurrentResolutionHeight(25);

    MonitorPlot monitorPlot_0 = ((MonitorPlot) simulation_0.getPlotManager().getPlot("drag Monitor \u7ED8\u56FE"));

    PlotUpdate plotUpdate_0 = monitorPlot_0.getPlotUpdate();

    HardcopyProperties hardcopyProperties_1 = plotUpdate_0.getHardcopyProperties();

    hardcopyProperties_1.setCurrentResolutionWidth(1033);

    hardcopyProperties_1.setCurrentResolutionHeight(483);

    hardcopyProperties_2.setCurrentResolutionWidth(1031);

    hardcopyProperties_2.setCurrentResolutionHeight(482);

    scene_1.resetCamera();

    CurrentView currentView_1 = scene_1.getCurrentView();

    simulation_0.getSceneManager().createGeometryScene("\u7F51\u683C\u573A\u666F", "\u8F6E\u5ED3", "\u7F51\u683C", 3);

    Scene scene_2 = simulation_0.getSceneManager().getScene("\u7F51\u683C\u573A\u666F 2");

    scene_2.initializeAndWait();

    SceneUpdate sceneUpdate_2 = scene_2.getSceneUpdate();

    HardcopyProperties hardcopyProperties_3 = sceneUpdate_2.getHardcopyProperties();

    hardcopyProperties_3.setCurrentResolutionWidth(25);

    hardcopyProperties_3.setCurrentResolutionHeight(25);

    hardcopyProperties_2.setCurrentResolutionWidth(1033);

    hardcopyProperties_2.setCurrentResolutionHeight(483);

    hardcopyProperties_3.setCurrentResolutionWidth(1031);

    hardcopyProperties_3.setCurrentResolutionHeight(482);

    scene_2.resetCamera();

    PartDisplayer partDisplayer_0 = ((PartDisplayer) scene_2.getDisplayerManager().getObject("\u7F51\u683C 1"));

    partDisplayer_0.getInputParts().setQuery(null);

    Region region_1 = simulation_0.getRegionManager().getRegion("domain");

    Boundary boundary_10 = region_1.getBoundaryManager().getBoundary("body.part_01_Body_Open_EngineBayFlow.Surface");


    Boundary boundary_14 = region_1.getBoundaryManager().getBoundary("body.part_02_UB_EngineBayFlow.Surface");

    Boundary boundary_15 = region_1.getBoundaryManager().getBoundary("body.part_03_"+back+".Surface");

    Boundary boundary_17 = region_1.getBoundaryManager()
        .getBoundary("body.part_04_ExhaustSystem_EngineBayFlow.Surface");

    Boundary boundary_18 = region_1.getBoundaryManager().getBoundary("body.part_07_Mirror.Surface");

    Boundary boundary_20 = region_1.getBoundaryManager()
        .getBoundary("body.part_08_EngineBayTrim_EngineBayFlow.Surface");

    Boundary boundary_21 = region_1.getBoundaryManager()
        .getBoundary("body.part_09_EngineAndGearbox_EngineBayFlow.Surface");

    Boundary boundary_22 = region_1.getBoundaryManager()
        .getBoundary("body.part_10_FrontGrills_EngineBayFlow.Surface");

    Boundary boundary_0 = region_1.getBoundaryManager().getBoundary("Porosity_EngineBayFlow.Surface");

    InterfaceBoundary interfaceBoundary_0 = ((InterfaceBoundary) region_1.getBoundaryManager()
        .getBoundary("Porosity_EngineBayFlow.Surface [\u4EA4\u754C\u9762 1]"));

    Boundary boundary_6 = region_1.getBoundaryManager().getBoundary("Wheels_Front.Surface");


    Boundary boundary_8 = region_1.getBoundaryManager().getBoundary("Wheels_Rear.Surface");


    partDisplayer_0.getInputParts().setObjects(boundary_10, boundary_14,
        boundary_15, boundary_17, boundary_18, boundary_20, boundary_21, boundary_22,
        boundary_0, interfaceBoundary_0, boundary_6, boundary_8);

    CurrentView currentView_2 = scene_2.getCurrentView();

    simulation_0.getSceneManager().createGeometryScene("\u7F51\u683C\u573A\u666F", "\u8F6E\u5ED3", "\u7F51\u683C", 3);

    Scene scene_3 = simulation_0.getSceneManager().getScene("\u7F51\u683C\u573A\u666F 3");

    scene_3.initializeAndWait();

    SceneUpdate sceneUpdate_3 = scene_3.getSceneUpdate();

    HardcopyProperties hardcopyProperties_4 = sceneUpdate_3.getHardcopyProperties();

    hardcopyProperties_4.setCurrentResolutionWidth(25);

    hardcopyProperties_4.setCurrentResolutionHeight(25);

    hardcopyProperties_3.setCurrentResolutionWidth(1033);

    hardcopyProperties_3.setCurrentResolutionHeight(483);

    hardcopyProperties_4.setCurrentResolutionWidth(1031);

    hardcopyProperties_4.setCurrentResolutionHeight(482);

    scene_3.resetCamera();

    CurrentView currentView_3 = scene_3.getCurrentView();

    currentView_3.setInput(new DoubleVector(new double[] { 6.052373311401389, 0.560133652021094, 2.9430717588297917 }),
        new DoubleVector(new double[] { 1.3581122765615206, -49.64508550689904, 27.372388093736816 }),
        new DoubleVector(new double[] { -0.05227758763066146, 0.4408901156928212, 0.8960373651337805 }),
        14.501702015860268, 0, 30.0);

    PartDisplayer partDisplayer_1 = ((PartDisplayer) scene_3.getDisplayerManager().getObject("\u7F51\u683C 1"));

    partDisplayer_1.getInputParts().setQuery(null);

    PlaneSection planeSection_0 = ((PlaneSection) simulation_0.getPartManager().getObject("\u5E73\u9762\u622A\u9762"));

    partDisplayer_1.getInputParts().setObjects(planeSection_0);

    currentView_3.setInput(new DoubleVector(new double[] { 2.31038626304692, 2.446407420895909, 3.6904580790853565 }),
        new DoubleVector(new double[] { -1.7294062878149261, -40.75925772369496, 24.713867177972727 }),
        new DoubleVector(new double[] { -0.05227758763066146, 0.4408901156928212, 0.8960373651337805 }),
        14.501702015860268, 0, 30.0);

    currentView_3.setInput(new DoubleVector(new double[] { 1.1311393582539626, 0.8433387397253895, 3.314421191808286 }),
        new DoubleVector(new double[] { -1.4482496562990614, -26.74328028521409, 16.737771474127932 }),
        new DoubleVector(new double[] { -0.05227758763066146, 0.4408901156928212, 0.8960373651337805 }),
        14.501702015860268, 0, 30.0);

    hardcopyProperties_1.setCurrentResolutionWidth(1031);

    hardcopyProperties_1.setCurrentResolutionHeight(482);

    currentView_3.setInput(
        new DoubleVector(new double[] { 11.255551890201044, 1.4187758540181576, 1.4334408170813653 }),
        new DoubleVector(new double[] { 5.446990036603693, -60.70391367978748, 31.661669268642953 }),
        new DoubleVector(new double[] { -0.05227758763066146, 0.4408901156928212, 0.8960373651337805 }),
        14.501702015860268, 0, 30.0);

    hardcopyProperties_2.setCurrentResolutionWidth(1031);

    hardcopyProperties_2.setCurrentResolutionHeight(482);

    hardcopyProperties_3.setCurrentResolutionWidth(1031);

    hardcopyProperties_3.setCurrentResolutionHeight(482);

    simulation_0.getSceneManager().createScalarScene("\u6807\u91CF\u573A\u666F", "\u8F6E\u5ED3", "\u6807\u91CF");

    Scene scene_4 = simulation_0.getSceneManager().getScene("\u6807\u91CF\u573A\u666F 1");

    scene_4.initializeAndWait();

    ScalarDisplayer scalarDisplayer_0 = ((ScalarDisplayer) scene_4.getDisplayerManager().getObject("\u6807\u91CF 1"));

    Legend legend_0 = scalarDisplayer_0.getLegend();

    PredefinedLookupTable predefinedLookupTable_0 = ((PredefinedLookupTable) simulation_0.get(LookupTableManager.class)
        .getObject("blue-yellow-red"));

    legend_0.setLookupTable(predefinedLookupTable_0);

    SceneUpdate sceneUpdate_4 = scene_4.getSceneUpdate();

    HardcopyProperties hardcopyProperties_5 = sceneUpdate_4.getHardcopyProperties();

    hardcopyProperties_5.setCurrentResolutionWidth(25);

    hardcopyProperties_5.setCurrentResolutionHeight(25);

    hardcopyProperties_3.setCurrentResolutionWidth(1033);

    hardcopyProperties_3.setCurrentResolutionHeight(483);

    hardcopyProperties_4.setCurrentResolutionWidth(1033);

    hardcopyProperties_4.setCurrentResolutionHeight(483);

    hardcopyProperties_5.setCurrentResolutionWidth(1031);

    hardcopyProperties_5.setCurrentResolutionHeight(482);

    scene_4.resetCamera();

    scalarDisplayer_0.getInputParts().setQuery(null);

    scalarDisplayer_0.getInputParts().setObjects(boundary_10, boundary_14,
        boundary_15, boundary_17, boundary_18, boundary_20, boundary_21, boundary_22,
        boundary_0, interfaceBoundary_0, boundary_6, boundary_8);

    CurrentView currentView_4 = scene_4.getCurrentView();

    PrimitiveFieldFunction primitiveFieldFunction_0 = ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager()
        .getFunction("Pressure"));

    scalarDisplayer_0.getScalarDisplayQuantity().setFieldFunction(primitiveFieldFunction_0);

    hardcopyProperties_4.setCurrentResolutionWidth(1032);

    hardcopyProperties_4.setCurrentResolutionWidth(1031);

    hardcopyProperties_4.setCurrentResolutionHeight(482);

    hardcopyProperties_3.setCurrentResolutionWidth(1031);

    hardcopyProperties_3.setCurrentResolutionHeight(482);

    simulation_0.getSceneManager().createScalarScene("\u6807\u91CF\u573A\u666F", "\u8F6E\u5ED3", "\u6807\u91CF");

    Scene scene_5 = simulation_0.getSceneManager().getScene("\u6807\u91CF\u573A\u666F 2");

    scene_5.initializeAndWait();

    ScalarDisplayer scalarDisplayer_1 = ((ScalarDisplayer) scene_5.getDisplayerManager().getObject("\u6807\u91CF 1"));

    Legend legend_1 = scalarDisplayer_1.getLegend();

    legend_1.setLookupTable(predefinedLookupTable_0);

    SceneUpdate sceneUpdate_5 = scene_5.getSceneUpdate();

    HardcopyProperties hardcopyProperties_6 = sceneUpdate_5.getHardcopyProperties();

    hardcopyProperties_6.setCurrentResolutionWidth(25);

    hardcopyProperties_6.setCurrentResolutionHeight(25);

    hardcopyProperties_3.setCurrentResolutionWidth(1033);

    hardcopyProperties_3.setCurrentResolutionHeight(483);

    hardcopyProperties_4.setCurrentResolutionWidth(1033);

    hardcopyProperties_4.setCurrentResolutionHeight(483);

    hardcopyProperties_6.setCurrentResolutionWidth(1031);

    hardcopyProperties_6.setCurrentResolutionHeight(482);

    scene_5.resetCamera();

    PartDisplayer partDisplayer_2 = ((PartDisplayer) scene_5.getDisplayerManager().getObject("\u8F6E\u5ED3 1"));

    partDisplayer_2.setOutline(false);
  }

  public void executeARF(String dir) {

    // Report : Frontal Area
    
    Simulation simulation_0 = 
      getActiveSimulation();

    FrontalAreaReport frontalAreaReport_1 = 
      simulation_0.getReportManager().create("star.vis.FrontalAreaReport");

    Units units_0 = 
      ((Units) simulation_0.getUnitsManager().getObject("m"));

    frontalAreaReport_1.getNormalCoordinate().setCoordinate(units_0, units_0, units_0, new DoubleVector(new double[] {1.0, 0.0, 0.0}));

    frontalAreaReport_1.getParts().setQuery(null);

    Region region_0 = 
      simulation_0.getRegionManager().getRegion("domain");

    Boundary boundary_5 = 
      region_0.getBoundaryManager().getBoundary("body.part_01_Body_Open_EngineBayFlow.Surface");

    Boundary boundary_6 = 
      region_0.getBoundaryManager().getBoundary("body.part_02_UB_EngineBayFlow.Surface");

    Boundary boundary_7 = 
      region_0.getBoundaryManager().getBoundary("body.part_03_Estate.Surface");

    Boundary boundary_8 = 
      region_0.getBoundaryManager().getBoundary("body.part_04_ExhaustSystem_EngineBayFlow.Surface");

    Boundary boundary_9 = 
      region_0.getBoundaryManager().getBoundary("body.part_07_Mirror.Surface");

    Boundary boundary_10 = 
      region_0.getBoundaryManager().getBoundary("body.part_08_EngineBayTrim_EngineBayFlow.Surface");

    Boundary boundary_11 = 
      region_0.getBoundaryManager().getBoundary("body.part_09_EngineAndGearbox_EngineBayFlow.Surface");

    Boundary boundary_12 = 
      region_0.getBoundaryManager().getBoundary("body.part_10_FrontGrills_EngineBayFlow.Surface");

    Boundary boundary_13 = 
      region_0.getBoundaryManager().getBoundary("Porosity_EngineBayFlow.Surface");

    Boundary boundary_14 = 
      region_0.getBoundaryManager().getBoundary("Wheels_Front.Surface");

    Boundary boundary_15 = 
      region_0.getBoundaryManager().getBoundary("Wheels_Rear.Surface");

    frontalAreaReport_1.getParts().setObjects(boundary_5, boundary_6, boundary_7, boundary_8, boundary_9, boundary_10, boundary_11, boundary_12, boundary_13, boundary_14, boundary_15);

    frontalAreaReport_1.printReport();

    ARF = frontalAreaReport_1.getReportMonitorValue();

    
    String filename = "info.txt";

    String filePath = dir + "/" + filename;

    try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath, true))) {
        writer.write("frontalArea(m^2): ");
        writer.write(Double.toString(ARF));
        writer.newLine(); 

        writer.write("ReferenceDensity(kg/m^3): ");
        writer.write(Double.toString(airDensity));
        writer.newLine(); 

        writer.write("ReferenceVelocity(m/s): ");
        writer.write(Double.toString(inlet_velocity/3.6));
        writer.newLine(); 
    } catch (IOException e) {
        e.printStackTrace();
    }

  }

  private void executeCD(String dir) {

 
    Simulation simulation_0 = 
      getActiveSimulation();

    ForceCoefficientReport forceCoefficientReport_1 = 
      simulation_0.getReportManager().create("star.flow.ForceCoefficientReport");

    Units units_1 = 
      ((Units) simulation_0.getUnitsManager().getObject("kg/m^3"));

    forceCoefficientReport_1.getReferenceDensity().setValueAndUnits(airDensity, units_1);

    Units units_2 = 
      ((Units) simulation_0.getUnitsManager().getObject("kph"));

    forceCoefficientReport_1.getReferenceVelocity().setValueAndUnits(inlet_velocity, units_2);

    Units units_3 = 
      ((Units) simulation_0.getUnitsManager().getObject("m^2"));

    forceCoefficientReport_1.getReferenceArea().setValueAndUnits(ARF, units_3);

    forceCoefficientReport_1.getParts().setQuery(null);

    Region region_0 = 
      simulation_0.getRegionManager().getRegion("domain");

    Boundary boundary_0 = 
      region_0.getBoundaryManager().getBoundary("body.part_01_Body_Open_EngineBayFlow.Surface");

    Boundary boundary_1 = 
      region_0.getBoundaryManager().getBoundary("body.part_02_UB_EngineBayFlow.Surface");

    Boundary boundary_2 = 
      region_0.getBoundaryManager().getBoundary("body.part_03_"+back+".Surface");

    Boundary boundary_3 = 
      region_0.getBoundaryManager().getBoundary("body.part_04_ExhaustSystem_EngineBayFlow.Surface");

    Boundary boundary_4 = 
      region_0.getBoundaryManager().getBoundary("body.part_07_Mirror.Surface");

    Boundary boundary_5 = 
      region_0.getBoundaryManager().getBoundary("body.part_08_EngineBayTrim_EngineBayFlow.Surface");

    Boundary boundary_6 = 
      region_0.getBoundaryManager().getBoundary("body.part_09_EngineAndGearbox_EngineBayFlow.Surface");

    Boundary boundary_7 = 
      region_0.getBoundaryManager().getBoundary("body.part_10_FrontGrills_EngineBayFlow.Surface");

    Boundary boundary_8 = 
      region_0.getBoundaryManager().getBoundary("Porosity_EngineBayFlow.Surface");

    Boundary boundary_9 = 
      region_0.getBoundaryManager().getBoundary("Wheels_Front.Surface");

    Boundary boundary_10 = 
      region_0.getBoundaryManager().getBoundary("Wheels_Rear.Surface");

    forceCoefficientReport_1.getParts().setObjects(boundary_0, boundary_1, boundary_2, boundary_3, boundary_4, boundary_5, boundary_6, boundary_7, boundary_8, boundary_9, boundary_10);

    forceCoefficientReport_1.printReport();

    simulation_0.getMonitorManager().createMonitorAndPlot(new NeoObjectVector(new Object[] {forceCoefficientReport_1}), true, "%1$s \u7ED8\u56FE");

    ReportMonitor reportMonitor_1 = 
      ((ReportMonitor) simulation_0.getMonitorManager().getMonitor("\u529B\u7CFB\u6570 1 Monitor"));

    MonitorPlot monitorPlot_2 = 
      simulation_0.getPlotManager().createMonitorPlot(new NeoObjectVector(new Object[] {reportMonitor_1}), "\u529B\u7CFB\u6570 1 Monitor \u7ED8\u56FE");

    monitorPlot_2.openInteractive();

    PlotUpdate plotUpdate_2 = 
      monitorPlot_2.getPlotUpdate();

    HardcopyProperties hardcopyProperties_6 = 
      plotUpdate_2.getHardcopyProperties();

    hardcopyProperties_6.setCurrentResolutionWidth(25);

    hardcopyProperties_6.setCurrentResolutionHeight(25);

    Scene scene_2 = 
      simulation_0.getSceneManager().getScene("\u51E0\u4F55\u573A\u666F 1");

    SceneUpdate sceneUpdate_2 = 
      scene_2.getSceneUpdate();

    HardcopyProperties hardcopyProperties_3 = 
      sceneUpdate_2.getHardcopyProperties();

    hardcopyProperties_3.setCurrentResolutionWidth(1532);

    hardcopyProperties_3.setCurrentResolutionHeight(549);

    MonitorPlot monitorPlot_0 = 
      ((MonitorPlot) simulation_0.getPlotManager().getPlot("drag Monitor \u7ED8\u56FE"));

    PlotUpdate plotUpdate_0 = 
      monitorPlot_0.getPlotUpdate();

    HardcopyProperties hardcopyProperties_1 = 
      plotUpdate_0.getHardcopyProperties();

    hardcopyProperties_1.setCurrentResolutionWidth(1532);

    hardcopyProperties_1.setCurrentResolutionHeight(549);

    hardcopyProperties_6.setCurrentResolutionWidth(1530);

    hardcopyProperties_6.setCurrentResolutionHeight(548);
  }


  private void setConstantDensityProperty() {

    Simulation simulation_0 = 
      getActiveSimulation();

    PhysicsContinuum physicsContinuum_0 = 
      ((PhysicsContinuum) simulation_0.getContinuumManager().getContinuum("\u7269\u7406 1"));

    SingleComponentGasModel singleComponentGasModel_0 = 
      physicsContinuum_0.getModelManager().getModel(SingleComponentGasModel.class);

    Gas gas_0 = 
      ((Gas) singleComponentGasModel_0.getMaterial());

    ConstantMaterialPropertyMethod constantMaterialPropertyMethod_0 = 
      ((ConstantMaterialPropertyMethod) gas_0.getMaterialProperties().getMaterialProperty(ConstantDensityProperty.class).getMethod());

    Units units_1 = 
      ((Units) simulation_0.getUnitsManager().getObject("kg/m^3"));

    constantMaterialPropertyMethod_0.getQuantity().setValueAndUnits(airDensity, units_1);
  }


}
