import os
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
import numpy as np
from scipy import stats
from scipy.spatial import distance
import csv
from pathlib import Path
from vtk.util.numpy_support import vtk_to_numpy


import SpineLib
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


    def loadModel(self, dirpath):
        for file in os.listdir(dirpath):
            filepath = os.path.join(dirpath, file)
            if file.endswith(".stl"):
                slicer.util.loadModel(filepath)
            if file.endswith(".json"):
                slicer.util.loadMarkups(filepath)
            if file.endswith(".h5"):
                slicer.util.loadTransform(filepath)


    # def run(self, directory, clAngle, tkAngle, llAngle, cAngle, ivdHeight):
    def run(self):
        print('Running Logic')

    

            


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
        symPlaneLogic = SymmetryPlaneLogic()
        symPlaneLogic.run()

        self.delayDisplay("Starting the test")
        # load segmentation models
        scriptPath = os.path.dirname(os.path.abspath(__file__))
        # get parent directory
        parentFolder = str(Path(scriptPath).parent).replace('SlicerPlugins','datasets')
        print(f'{parentFolder}')
        sawboneDirectory = os.path.join(parentFolder, "L1")
        sawboneFiles = slicer.util.getFilesInDirectory(sawboneDirectory)
        sawboneModelNodes = []

        for file in sawboneFiles:
            if file.endswith(".stl") or file.endswith(".obj"):
                node = slicer.util.loadModel(file)
                sawboneModelNodes.append(node)
                polydata = node.GetPolyData()
                centerOfMass = conv.calc_center_of_mass(polydata)
                points = polydata.GetPoints()
                array = points.GetData()
                points = vtk_to_numpy(array)
                lineNodeNames = ['e1', 'e2', 'e3']
                lineStartPos = np.asarray(centerOfMass)
                # Get initial symmetry plane P0 using PCA
                eigenvects = conv.pca_eigenvectors(points)
                i = 0
                for lineNodeName in lineNodeNames:
                    e = eigenvects[i]
                    print(e)
                    lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", lineNodeName)
                    
                    print(f'lineStartPos: {lineStartPos}')
                    lineEndPos = lineStartPos + (e*100)
                    print(f'lineEndPos: {lineEndPos}')
                    lineNode.AddControlPointWorld(lineStartPos, f'start{i}')
                    lineNode.AddControlPointWorld(lineEndPos, f'end{i}')
                    i+=1
                    #lineDirectionVector = (lineEndPos-lineStartPos)/np.linalg.norm(lineEndPos-lineStartPos)
                    

                # Compute the covariance matrix and principal components without centering



        self.delayDisplay('Test passed!')