import os
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
import numpy as np
from scipy import stats
from scipy.spatial import distance
from scipy.optimize import least_squares
import csv
from pathlib import Path
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import open3d as o3d

from scipy.linalg import lstsq
from skspatial.objects import Plane, Points

import SpineLib
from  SpineLib import SlicerTools
import vtk_convenience as conv

#https://github.com/Caetox/SpineLib.git
# Create spine from induvidual sawbones, with given cervical lordosis, thoracic kyphosis and lumbar lordosis angles
#


class SymmetryPlane(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Symmetry Plane"
        self.parent.categories = ["VisSim Morphology Tools"]
        self.parent.dependencies = []
        self.parent.contributors = []
        self.parent.helpText = """TODO"""
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = """TODO"""  # replace with organization, grant and thanks.

#
# SymmetryPlaneWidget
#

class SymmetryPlaneWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        scriptPath = os.path.dirname(os.path.abspath(__file__))
    
    # enddef setup



    def cleanup(self):
        pass



#
# SymmetryPlaneLogic
#

class SymmetryPlaneLogic(ScriptedLoadableModuleLogic):

    def __init__(self, pathToFiles):
        self.vertebraFiles = pathToFiles
        #if we need some parameters, set them here


    def updatePlane(self, vectors):

        angles = []

        # calculate vertebra angles
        for index in range(0,len(vectors)-1):
            angleRad = vtk.vtkMath.AngleBetweenVectors(vectors[index], vectors[index+1])
            angleDeg = vtk.vtkMath.DegreesFromRadians(angleRad)
            angle = "{0:0.3f}".format(angleDeg)
            angles.append(angle)

        # calculate lumbar lordosis angle
        angleRad = vtk.vtkMath.AngleBetweenVectors(vectors[0], vectors[5])
        angleDeg = vtk.vtkMath.DegreesFromRadians(angleRad)
        angle = "{0:0.3f}".format(angleDeg)
        angles.append(angle)

        return angles

    def calcInitalPlane(self, points, centerOfMass):
        '''
        points: nparray. Takes numpy array and calculates the pca axes to define the initial symmetry plane (sagittal plane in our case)
        centerOfMass: tuple. It is used to display the axes
        returns: inital sagittal plane
        '''
        lineNodeNames = ['e1', 'e2', 'e3']
        lineNodes = []
        lineStartPos = np.asarray(centerOfMass)
        eigenvects = conv.pca_eigenvectors(points)
        i = 0
        planeNormal = eigenvects[1]

        for lineNodeName in lineNodeNames:
            e = eigenvects[i]
            #we take e2 as normal to sagittal plane
            lineEndPos = lineStartPos + (e*100)
            lineNode = SlicerTools.markupsLineNode(lineNodeName, lineStartPos, lineEndPos)
            lineNodes.append(lineNode)
            i +=1 

                    
        # Get initial symmetry plane P0 using PCA
        planeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", "initPlane_P0")
        planeNode.SetPlaneType(slicer.vtkMRMLMarkupsPlaneNode.PlaneTypePointNormal)
        print(centerOfMass)
        planeNode.SetCenter(centerOfMass)
        planeNode.SetNormal(planeNormal)

        return planeNode


    def cutVertebraModelWithPlane(self, planeNode, vertModelNode):
        
        planeCutTool = slicer.vtkSlicerDynamicModelerPlaneCutTool()
          
        outputModel_0 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        outputModel_1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        
        dynamicModelerNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLDynamicModelerNode")
        dynamicModelerNode.SetToolName("Plane cut")
        dynamicModelerNode.SetNodeReferenceID("PlaneCut.InputModel", vertModelNode.GetID())
        dynamicModelerNode.SetNodeReferenceID("PlaneCut.InputPlane", planeNode.GetID())
        dynamicModelerNode.SetNodeReferenceID("PlaneCut.OutputPositiveModel", outputModel_0.GetID())
        dynamicModelerNode.SetNodeReferenceID("PlaneCut.OutputNegativeModel", outputModel_1.GetID())
        slicer.modules.dynamicmodeler.logic().RunDynamicModelerTool(dynamicModelerNode)

        return outputModel_0, outputModel_1


    def mirrorVertebraModelUponPlane(self, planeNode, vertModelNode, i):
        print(f'Using plane node: {planeNode.GetName()}')
        #we can use this output to print the attributes for this synamic modeler
        #planeMirrorTool = slicer.vtkSlicerDynamicModelerMirrorTool()
        
        #for i in range(planeMirrorTool.GetNumberOfOutputNodes()):
        #    print(planeMirrorTool.GetNthOutputNodeReferenceRole(i))

        outputMirrored = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        outputMirrored.SetName(f'Mirrored_{i}')
        dynamicModelerNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLDynamicModelerNode")
        dynamicModelerNode.SetToolName("Mirror")
        dynamicModelerNode.SetNodeReferenceID("Mirror.InputModel", vertModelNode.GetID())
        dynamicModelerNode.SetNodeReferenceID("Mirror.InputPlane", planeNode.GetID())
        dynamicModelerNode.SetNodeReferenceID("Mirror.OutputModel", outputMirrored.GetID())
        slicer.modules.dynamicmodeler.logic().RunDynamicModelerTool(dynamicModelerNode)

        return outputMirrored


    def loadModel(self, dirpath):
        for file in os.listdir(dirpath):
            filepath = os.path.join(dirpath, file)
            if file.endswith(".stl"):
                slicer.util.loadModel(filepath)
            if file.endswith(".json"):
                slicer.util.loadMarkups(filepath)
            if file.endswith(".h5"):
                slicer.util.loadTransform(filepath)


    def getMiddlePoints(self, originalPoints, mirroredPoints, viz=True):
        
        if originalPoints.shape[0] > 10000:
            inds = np.random.choice(originalPoints.shape[0], 10000)
            originalPoints = np.take(originalPoints,inds, axis=0)
            mirroredPoints = np.take(mirroredPoints,inds, axis=0)

        middlepoints = (originalPoints + mirroredPoints)/2.0

        if viz:

            vtk_points = vtk.vtkPoints()
            vtk_points.SetData(vtk.util.numpy_support.numpy_to_vtk(middlepoints))

            # Create the vtkPolyData object.
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(vtk_points)

            # Create the vtkSphereSource object.
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(2.0)

            # Create the vtkGlyph3D object.
            glyph = vtk.vtkGlyph3D()
            glyph.SetInputData(polydata)
            glyph.SetSourceConnection(sphere.GetOutputPort())


            pointCloudModelNode = slicer.modules.models.logic().AddModel(glyph.GetOutputPort())

        return middlepoints


        
        
    def fitPlaneLeastSquered(self, points, centerOfMass,i):
        plane = Plane.best_fit(Points(points))
        normal = plane.normal.round(3)
        print(plane.normal.round(3))
        print(type(normal))

        # Get initial symmetry plane P0 using PCA
        planeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", f"plane_P{i}")
        planeNode.SetPlaneType(slicer.vtkMRMLMarkupsPlaneNode.PlaneTypePointNormal)
        planeNode.SetCenter(centerOfMass)
        planeNode.SetNormal(normal)

        return planeNode


    def fitPlaneSVD(self, points, centerOfMass, i):

        A = np.c_[points[:, 0:2], np.ones(points.shape[0])]
        print(A.shape)
        b = points[:, 0]
        print(b.shape)
        
        p, res, rnk, s = lstsq(A, b)
        print(p, res, rnk, s )
        print("Plane coefficients [A, B, C]:", p)

        # Get initial symmetry plane P0 using PCA
        planeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", f"plane_P{i}")
        planeNode.SetPlaneType(slicer.vtkMRMLMarkupsPlaneNode.PlaneTypePointNormal)
        planeNode.SetCenter(centerOfMass)
        planeNode.SetNormal(p)

        return planeNode



    def evaluateRegistrationICP(self, originalPoints, mirroredPoints):
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(originalPoints)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(mirroredPoints)
        threshold = 10000
        evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold)

        print(f'evaluation.fitness={evaluation.fitness}, evaluation.inlier_rmse={evaluation.inlier_rmse}')
        return evaluation.fitness, evaluation.inlier_rmse


    def registerWithVanilaICP(self, originalPoints, mirroredPoints):
        threshold = 100000 #mm
        trans_init = np.identity(4)
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(originalPoints)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(mirroredPoints)
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
        icp = reg_p2p.transformation
        transformationMatrix = vtk.vtkMatrix4x4()

        # Populate the VTK matrix with values from the NumPy matrix
        for i in range(4):
            for j in range(4):
                transformationMatrix.SetElement(i, j, icp[i, j])

        return transformationMatrix

    def approximateVoramenCylinder(self):
        pass


    # def run(self, directory, clAngle, tkAngle, llAngle, cAngle, ivdHeight):
    def run(self):
        print('Running Logic')
        sawboneModelNodes = []
        transformationMatrix = None
        for file in self.vertebraFiles:
            if file.endswith(".stl") or file.endswith(".obj"):
                vertebraModelNode = slicer.util.loadModel(file)
                sawboneModelNodes.append(vertebraModelNode)
                polydata = vertebraModelNode.GetPolyData()
                centerOfMass = conv.calc_center_of_mass(polydata)

                centerOfMassNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "centerOfMass")
                centerOfMassNode.AddFiducial(centerOfMass[0],centerOfMass[1],centerOfMass[2])

                points = SlicerTools.pointsFromModelNode_asNumPy(vertebraModelNode)
                
                p0 = self.calcInitalPlane(points, centerOfMass)
                mirroredVertNode = self.mirrorVertebraModelUponPlane(p0, vertebraModelNode, 100)
                pFin = None
                for i in range(2):
                
                    mirroredModelPoints = SlicerTools.pointsFromModelNode_asNumPy(mirroredVertNode)
                    transformationMatrix = self.registerWithVanilaICP(mirroredModelPoints, points)
                    SpineLib.SlicerTools.transformOneObject(transformationMatrix, mirroredVertNode)
                    mirroredModelPoints = SlicerTools.pointsFromModelNode_asNumPy(mirroredVertNode)
                    self.evaluateRegistrationICP(mirroredModelPoints, points)
                    
                    middlePoints = self.getMiddlePoints(points, mirroredModelPoints)
                    p_i = self.fitPlaneLeastSquered(middlePoints, centerOfMass, i)
                    mirroredVertNode = self.mirrorVertebraModelUponPlane(p_i, vertebraModelNode, i)
                    pFin = p_i
                
                print(f'plane final {pFin.GetName()}')
                print(pFin.GetOrigin(), pFin.GetNormal())

                contour = conv.cut_plane(polydata,pFin.GetOrigin(), pFin.GetNormal())
                points = contour.GetPoints().GetData()
                
                
                print(vtk_to_numpy(points).shape)

                vtk_points = vtk.vtkPoints()
                vtk_points.SetData(vtk.util.numpy_support.numpy_to_vtk(vtk_to_numpy(points)))

                # Create the vtkPolyData object.
                polydata = vtk.vtkPolyData()
                polydata.SetPoints(vtk_points)

                # Create the vtkSphereSource object.
                sphere = vtk.vtkSphereSource()
                sphere.SetRadius(1.0)

                # Create the vtkGlyph3D object.
                glyph = vtk.vtkGlyph3D()
                glyph.SetInputData(polydata)
                glyph.SetSourceConnection(sphere.GetOutputPort())


                pointCloudModelNode = slicer.modules.models.logic().AddModel(glyph.GetOutputPort())

                lineNodeNames = ['e11', 'e21', 'e31']
                lineNodes = []
                lineStartPos = np.asarray(centerOfMass)
                eigenvects = conv.pca_eigenvectors(vtk_to_numpy(points))
                i = 0
                
                for lineNodeName in lineNodeNames:
                    e = eigenvects[i]
                    #we take e2 as normal topoints sagittal plane
                    lineEndPos = lineStartPos + (e*100)
                    lineNode = SlicerTools.markupsLineNode(lineNodeName, lineStartPos, lineEndPos)
                    lineNodes.append(lineNode)
                    i +=1 


                


class SymmetryPlaneTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_SymmetryPlane1()

    def test_SymmetryPlane1(self):
        

        self.delayDisplay("Starting the test")
        # load segmentation models
        scriptPath = os.path.dirname(os.path.abspath(__file__))
        # get parent directory
        parentFolder = str(Path(scriptPath).parent).replace('SlicerPlugins','datasets')
        print(f'{parentFolder}')
        sawboneDirectory = os.path.join(parentFolder, "L1")
        sawboneFiles = slicer.util.getFilesInDirectory(sawboneDirectory)
        
        symPlaneLogic = SymmetryPlaneLogic(sawboneFiles)
        symPlaneLogic.run()

        self.delayDisplay('Test passed!')