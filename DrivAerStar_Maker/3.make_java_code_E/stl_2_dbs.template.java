// Simcenter STAR-CCM+ macro: p32dbs.java
// Written by Simcenter STAR-CCM+ 18.06.007
package macro;

import java.util.*;

import star.common.*;
import star.base.neo.*;
import star.vis.*;
import star.meshing.*;
import java.io.File;

public class stl_2_dbs extends StarMacro {

  public void execute() {
    String formattedNumber = "<<<id>>>";
    String dirin =  "<<<dir>>>/";
    String inputFile = dirin + "part_05_Wheels_Front.stl";
    String dirout =  "<<<dir>>>/";
    String outputFile = dirout + "Porosity_EngineBayFlow.dbs";

    File input = new File(inputFile);
    File output = new File(outputFile);

    if (input.exists() &&!output.exists()) {
        execute0(formattedNumber);
    }
        
  }

  private void execute0(String dirid) {
    String dirin =  "<<<dir>>>/";
    String dirout =  "<<<dir>>>/";

    Simulation simulation_0 = 
      getActiveSimulation();

    Units units_0 = 
      simulation_0.getUnitsManager().getPreferredUnits(Dimensions.Builder().length(1).build());

    PartImportManager partImportManager_0 = 
      simulation_0.get(PartImportManager.class);

    Units units_1 = 
      ((Units) simulation_0.getUnitsManager().getObject("mm"));

    partImportManager_0.importStlParts(new StringVector(new String[] {resolvePath(dirin+"part_05_Wheels_Front.stl"), resolvePath(dirin+"part_06_Wheels_Rear.stl"), resolvePath(dirin+"part_11B_PressureLoss_Porosity_EngineBayFlow.stl")}), "OneSurfacePerPatch", "OnePartPerFile", units_1, true, 1.0E-5, false, false);

    simulation_0.getSceneManager().createGeometryScene("Geometry Scene", "Outline", "Surface", 1, null);

    Scene scene_0 = 
      simulation_0.getSceneManager().getScene("Geometry Scene 1");

    scene_0.initializeAndWait();

    SceneUpdate sceneUpdate_0 = 
      scene_0.getSceneUpdate();

    HardcopyProperties hardcopyProperties_0 = 
      sceneUpdate_0.getHardcopyProperties();

    hardcopyProperties_0.setCurrentResolutionWidth(1530);

    hardcopyProperties_0.setCurrentResolutionHeight(548);

    scene_0.resetCamera();

    CurrentView currentView_0 = 
      scene_0.getCurrentView();

    currentView_0.setInput(new DoubleVector(new double[] {0.7292056430886691, 0.23221955892171647, -0.4086684202722335}), new DoubleVector(new double[] {0.7292056430886691, 0.23221955892171647, 4.719102291086793}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 1.984009289446951, 0, 30.0);

    currentView_0.setInput(new DoubleVector(new double[] {0.7292056430886691, 0.23221955892171647, -0.4086684202722335}), new DoubleVector(new double[] {0.7292056430886691, 0.23221955892171647, 4.719102291086793}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 1.984009289446951, 0, 30.0);

    currentView_0.setInput(new DoubleVector(new double[] {0.7292056430886691, 0.23221955892171647, -0.4086684202722335}), new DoubleVector(new double[] {0.7292056430886691, 0.23221955892171647, 4.719102291086793}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 1.984009289446951, 0, 30.0);

    currentView_0.setInput(new DoubleVector(new double[] {0.7292056430886691, 0.23221955892171647, -0.4086684202722335}), new DoubleVector(new double[] {0.7292056430886691, 0.23221955892171647, 4.719102291086793}), new DoubleVector(new double[] {0.0, 1.0, 0.0}), 1.984009289446951, 0, 30.0);

    currentView_0.setInput(new DoubleVector(new double[] {0.6765622514919123, -0.39578082092612465, 0.6181954990147187}), new DoubleVector(new double[] {0.2842383267110451, -5.075942305828593, 2.345597503979724}), new DoubleVector(new double[] {-0.05450754588673751, 0.3497615088395294, 0.9352517385045056}), 1.984009289446951, 0, 30.0);

    RootDescriptionSource rootDescriptionSource_0 = 
      simulation_0.get(SimulationMeshPartDescriptionSourceManager.class).getRootDescriptionSource();

    MeshPart meshPart_0 = 
      ((MeshPart) simulation_0.get(SimulationPartManager.class).getPart("part_05_Wheels_Front"));

    rootDescriptionSource_0.exportDbsPartDescriptions(new NeoObjectVector(new Object[] {meshPart_0}), resolvePath(dirout+"Wheels_Front.dbs"), 1, "");

    MeshPart meshPart_1 = 
      ((MeshPart) simulation_0.get(SimulationPartManager.class).getPart("part_06_Wheels_Rear"));

    rootDescriptionSource_0.exportDbsPartDescriptions(new NeoObjectVector(new Object[] {meshPart_1}), resolvePath(dirout+"Wheels_Rear.dbs"), 1, "");

    MeshPart meshPart_2 = 
      ((MeshPart) simulation_0.get(SimulationPartManager.class).getPart("part_11B_PressureLoss_Porosity_EngineBayFlow"));

    rootDescriptionSource_0.exportDbsPartDescriptions(new NeoObjectVector(new Object[] {meshPart_2}), resolvePath(dirout+"Porosity_EngineBayFlow.dbs"), 1, "");

    simulation_0.get(SimulationPartManager.class).removeObjects(meshPart_0, meshPart_1, meshPart_2);
  }
}
