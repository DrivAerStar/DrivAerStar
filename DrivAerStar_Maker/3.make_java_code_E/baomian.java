// Simcenter STAR-CCM+ macro: baomian.java
// Written by Simcenter STAR-CCM+ 18.06.007
package macro;

import java.io.File;

import java.util.*;

import star.common.*;
import star.base.neo.*;
import star.vis.*;
import star.meshing.*;
import star.surfacewrapper.*;

public class baomian extends StarMacro {

  public void execute() {
    String formattedNumber = "<<<id>>>";
    String folderPath =  "<<<dir>>>/";
    String back =  "<<<back>>>";
    File folder = new File(folderPath);
    if (!folder.exists()) {
      boolean created = folder.mkdirs();
      if (created) {
        System.out.println("files " + formattedNumber + " win");
      } else {
        System.out.println("files " + formattedNumber + " error");
      }
    }
    String filePath = folderPath + "/body.dbs";
    File file = new File(filePath);
    if (file.exists()) {
      System.out.println(formattedNumber + " exists");
    } else {
      execute0();
    }
    
  }

private void execute0() {
    String dirin =  "<<<dir>>>/";
    String dirout =  "<<<dir>>>/";
    String back =  "<<<back>>>";
    Simulation simulation_0 = 
      getActiveSimulation();

    Units units_0 = 
      simulation_0.getUnitsManager().getPreferredUnits(Dimensions.Builder().length(1).build());

    PartImportManager partImportManager_0 = 
      simulation_0.get(PartImportManager.class);

    Units units_1 = 
      ((Units) simulation_0.getUnitsManager().getObject("mm"));

    partImportManager_0.importStlParts(new StringVector(new String[] {resolvePath(dirin+"part_01_Body_Open_EngineBayFlow.stl"), resolvePath(dirin+"part_02_UB_EngineBayFlow.stl"), resolvePath(dirin+"part_03_"+back+".stl"), resolvePath(dirin+"part_04_ExhaustSystem_EngineBayFlow.stl"), resolvePath(dirin+"part_07_Mirror.stl"), resolvePath(dirin+"part_08_EngineBayTrim_EngineBayFlow.stl"), resolvePath(dirin+"part_09_EngineAndGearbox_EngineBayFlow.stl"), resolvePath(dirin+"part_10_FrontGrills_EngineBayFlow.stl")}), "OneSurfacePerPatch", "OnePartPerFile", units_1, true, 1.0E-5, false, false);

    simulation_0.getSceneManager().createGeometryScene("Geometry Scene", "Outline", "Surface", 1, null);

    Scene scene_0 = 
      simulation_0.getSceneManager().getScene("Geometry Scene 1");

    scene_0.initializeAndWait();

    SceneUpdate sceneUpdate_0 = 
      scene_0.getSceneUpdate();

    HardcopyProperties hardcopyProperties_0 = 
      sceneUpdate_0.getHardcopyProperties();

    hardcopyProperties_0.setCurrentResolutionWidth(997);

    hardcopyProperties_0.setCurrentResolutionHeight(420);

    scene_0.resetCamera();

    MeshPart meshPart_0 = 
      ((MeshPart) simulation_0.get(SimulationPartManager.class).getPart("part_01_Body_Open_EngineBayFlow"));

    MeshPart meshPart_1 = 
      ((MeshPart) simulation_0.get(SimulationPartManager.class).getPart("part_02_UB_EngineBayFlow"));

    MeshPart meshPart_2 = 
      ((MeshPart) simulation_0.get(SimulationPartManager.class).getPart("part_03_"+back));

    MeshPart meshPart_3 = 
      ((MeshPart) simulation_0.get(SimulationPartManager.class).getPart("part_04_ExhaustSystem_EngineBayFlow"));

    MeshPart meshPart_4 = 
      ((MeshPart) simulation_0.get(SimulationPartManager.class).getPart("part_07_Mirror"));

    MeshPart meshPart_5 = 
      ((MeshPart) simulation_0.get(SimulationPartManager.class).getPart("part_08_EngineBayTrim_EngineBayFlow"));

    MeshPart meshPart_6 = 
      ((MeshPart) simulation_0.get(SimulationPartManager.class).getPart("part_09_EngineAndGearbox_EngineBayFlow"));

    MeshPart meshPart_7 = 
      ((MeshPart) simulation_0.get(SimulationPartManager.class).getPart("part_10_FrontGrills_EngineBayFlow"));

    SurfaceWrapperAutoMeshOperation surfaceWrapperAutoMeshOperation_0 = 
      (SurfaceWrapperAutoMeshOperation) simulation_0.get(MeshOperationManager.class).createSurfaceWrapperAutoMeshOperation(new NeoObjectVector(new Object[] {meshPart_0, meshPart_1, meshPart_2, meshPart_3, meshPart_4, meshPart_5, meshPart_6, meshPart_7}), "Surface Wrapper");

    CurrentView currentView_0 = 
      scene_0.getCurrentView();

    currentView_0.setInput(new DoubleVector(new double[] {1.5034964904785157, -2.1057128907209233E-6, 0.4528818244934083}), new DoubleVector(new double[] {1.5034964904785157, -2.1057128907209233E-6, 8.789766321212609}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 2.1577444845708724, 0, 30.0);

    hardcopyProperties_0.setCurrentResolutionWidth(921);

    surfaceWrapperAutoMeshOperation_0.getDefaultValues().get(BaseSize.class).setValueAndUnits(0.05, units_0);

    PartsTargetSurfaceSize partsTargetSurfaceSize_0 = 
      surfaceWrapperAutoMeshOperation_0.getDefaultValues().get(PartsTargetSurfaceSize.class);

    Units units_2 = 
      ((Units) simulation_0.getUnitsManager().getObject(""));

    partsTargetSurfaceSize_0.getRelativeSizeScalar().setValueAndUnits(10.0, units_2);

    PartsMinimumSurfaceSize partsMinimumSurfaceSize_0 = 
      surfaceWrapperAutoMeshOperation_0.getDefaultValues().get(PartsMinimumSurfaceSize.class);

    partsMinimumSurfaceSize_0.getRelativeSizeScalar().setValueAndUnits(5.0, units_2);

    GlobalVolumeOfInterest globalVolumeOfInterest_0 = 
      surfaceWrapperAutoMeshOperation_0.getDefaultValues().get(GlobalVolumeOfInterest.class);

    globalVolumeOfInterest_0.getVolumeOfInterestOption().setSelected(GlobalVolumeOfInterestOption.Type.EXTERNAL);

    SurfaceCustomMeshControl surfaceCustomMeshControl_0 = 
      surfaceWrapperAutoMeshOperation_0.getCustomMeshControls().createSurfaceControl();

    surfaceCustomMeshControl_0.getGeometryObjects().setQuery(null);

    surfaceCustomMeshControl_0.getGeometryObjects().setObjects(meshPart_4, meshPart_5, meshPart_6, meshPart_7);

    surfaceCustomMeshControl_0.getCustomConditions().get(PartsTargetSurfaceSizeOption.class).setSelected(PartsTargetSurfaceSizeOption.Type.CUSTOM);

    PartsTargetSurfaceSize partsTargetSurfaceSize_1 = 
      surfaceCustomMeshControl_0.getCustomValues().get(PartsTargetSurfaceSize.class);

    partsTargetSurfaceSize_1.getRelativeSizeScalar().setValueAndUnits(3.0, units_2);

    PartsTwoGroupContactPreventionSet partsTwoGroupContactPreventionSet_0 = 
      surfaceWrapperAutoMeshOperation_0.getContactPreventionSet().createPartsTwoGroupContactPreventionSet();

    partsTwoGroupContactPreventionSet_0.getPartSurfaceGroup1().setQuery(null);

    partsTwoGroupContactPreventionSet_0.getPartSurfaceGroup1().setObjects(meshPart_0, meshPart_1, meshPart_2, meshPart_3, meshPart_4, meshPart_7);

    partsTwoGroupContactPreventionSet_0.getPartSurfaceGroup2().setQuery(null);

    partsTwoGroupContactPreventionSet_0.getPartSurfaceGroup2().setObjects(meshPart_5, meshPart_6);

    surfaceWrapperAutoMeshOperation_0.execute();

    RootDescriptionSource rootDescriptionSource_0 = 
      simulation_0.get(SimulationMeshPartDescriptionSourceManager.class).getRootDescriptionSource();

    MeshOperationPart meshOperationPart_0 = 
      ((MeshOperationPart) simulation_0.get(SimulationPartManager.class).getPart("Surface Wrapper"));

    rootDescriptionSource_0.exportDbsPartDescriptions(new NeoObjectVector(new Object[] {meshOperationPart_0}), resolvePath(dirout+"body.dbs"), 1, "");

    simulation_0.get(SimulationPartManager.class).removeObjects(meshPart_0, meshPart_1, meshPart_2, meshPart_3, meshPart_4, meshPart_5, meshPart_6, meshPart_7, meshOperationPart_0);

    scene_0.closeInteractive();

    simulation_0.get(MeshOperationManager.class).removeObjects(surfaceWrapperAutoMeshOperation_0);

    simulation_0.getSceneManager().removeObjects(scene_0);
  }
}
