from __future__ import annotations
import os

import logging
import numpy as np
from copy import copy
from enum import IntEnum, unique, auto
from typing import Any, List, Tuple
from itertools import count
from dataclasses import dataclass
from qt import QTableWidgetItem
import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

# workaround for loading external py file in Slicer
import importlib.util
import sys

def from_module_import(module_name: str, *elements: Tuple[str]) -> Tuple[Any]:
    """ Import any module hosted inside "__file__/Resources/Scripts" by string name """
    module_file = f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(os.path.dirname(__file__), 'Resources', 'Scripts', module_file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return tuple(sys.modules[module_name].__dict__[el] for el in elements)

from_module_import("vtk_convenience")
Vector3D, *_ = from_module_import("vtk_convenience", "Vector3D")
Spine, Endplate, Body = from_module_import("morphology", "Spine", "Endplate", "Body")

#
# Dimensions
#
class Dimensions(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Dimensions"
        self.parent.categories = ["VisSim"]
        self.parent.dependencies = []
        self.parent.contributors = ["Vinzent Rittel (Research Group VisSim)"]
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This module calculates the endplate slope for a vertebra geometry.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This module is the result of the works of Dr. rer. nat. Sabine Bauer, Vinzent Rittel and MSc Ivanna Kramer.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#
def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # Dimensions1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='Dimensions',
        sampleName='Dimensions1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'Dimensions1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='Dimensions1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='Dimensions1'
    )

    # Dimensions2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='Dimensions',
        sampleName='Dimensions2',
        thumbnailFileName=os.path.join(iconsPath, 'Dimensions2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='Dimensions2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='Dimensions2'
    )


@unique
class DisplayMode(IntEnum):
    Vertebra = 0
    Body = auto()
    Slice = auto()

    @classmethod
    @property
    def all_modes(cls):
        return [item.name for item in cls]

#
# DimensionsWidget
#

class DimensionsWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/Dimensions.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = DimensionsLogic()

        self.mrmlHierarchy = slicer.mrmlScene.GetSubjectHierarchyNode()
        self.scene_item_id = self.mrmlHierarchy.GetSceneItemID()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(self.mrmlHierarchy, self.mrmlHierarchy.SubjectHierarchyItemModifiedEvent, self.updateInputSelector)
        self.updateInputSelector()

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputSelector.connect("currentTextChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputSelector.connect("currentTextChanged(vtkMRMLNode*)", self.clearUI)
        self.ui.displayModeComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.displayModeComboBox.connect("currentIndexChanged(int)", self.updateDisplayMode)
        self.ui.saveResultButton.connect("clicked(bool)", self.onSaveResults)
        self.ui.resultPathLineEdit.connect("currentPathChanged(QString)", self.enableSaveResultButton)

        self.rightVectorSection = self.DirectionVector("right", self)

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        self.initializeDisplayModeComboBox()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        if self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetParameter("InputGeometries"):
            folders = [child for child in self.logic.getChildren(self.scene_item_id) if self.logic.isItemAFolder(child)]
            if folders:
                self._parameterNode.SetParameter("InputGeometries", str(folders[0]))

        if not self._parameterNode.GetParameter("DisplayMode"):
            self.ui.displayModeComboBox.addItems(DisplayMode.all_modes)
            self.ui.displayModeComboBox.currentIndex = DisplayMode.VERTEBRA.value
            self._parameterNode.SetParameter("DisplayMode", str(DisplayMode.VERTEBRA.value))

        if not self._parameterNode.GetParameter("TopVertebra"):
            self.ui.resultVertebraSelector.clear()
            self.ui.resultVertebraSelector.addItems(Spine.VERTEBRAE)
            self._parameterNode.SetParameter("TopVertebra", "0")

    def clearUI(self, node=None):
        self.ui.displayModeComboBox.currentIndex = 0
        self.ui.displayModeComboBox.enabled = False

        self.ui.resultsCollapsibleButton.collapsed = True
        self.ui.resultsCollapsibleButton.enabled = False
        self.ui.resultVertebraSelector.clear()
        self.ui.resultVertebraSelector.addItems(Spine.VERTEBRAE)

        lvl1Nodes = self.logic.getChildren(self.mrmlHierarchy.GetSceneItemID())
        dissectionFolderPostfix = self.logic.dissectionFolderName("")
        dissectionFolders = [folder for folder in lvl1Nodes if dissectionFolderPostfix in self.mrmlHierarchy.GetItemName(folder)]
        for folder in dissectionFolders:
            self.mrmlHierarchy.RemoveItem(folder)

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateInputSelector(self, *args):
        folder_names = [
            self.mrmlHierarchy.GetItemName(item)
            for item in self.logic.getChildren(self.scene_item_id)
            if self.logic.isItemAFolder(item)
        ]

        self.ui.inputSelector.clear()
        self.ui.inputSelector.addItems(folder_names)
        self.ui.inputSelector.enabled = len(folder_names) > 0

    def updateDisplayMode(self, newDisplayMode):
        dissectionNodes = dict(zip(
            DisplayMode.all_modes,
            self.logic.dissectionFolderContent(self.selectedFolderName),
        ))
        if not dissectionNodes:
            return

        self.setVisibility(self.selectedFolderId(), isVisible=False)
        dissectionFolderId = self.logic.dissectionFolderId(self.selectedFolderName)
        if newDisplayMode == DisplayMode.Vertebra:
            self.setVisibility(
                self.mrmlHierarchy.GetItemChildWithName(dissectionFolderId, DisplayMode.Vertebra.name)
            )
            self.setVisibility(
                self.mrmlHierarchy.GetItemChildWithName(dissectionFolderId,
                DisplayMode.Body.name),
                isVisible=False,
            )
            self.setVisibility(
                self.mrmlHierarchy.GetItemChildWithName(dissectionFolderId,
                DisplayMode.Slice.name),
                isVisible=False,
            )
        elif newDisplayMode == DisplayMode.Body:
            self.setVisibility(
                self.mrmlHierarchy.GetItemChildWithName(dissectionFolderId,
                DisplayMode.Vertebra.name),
                isWireframe=True,
                opacity=0.5,
            )
            self.setVisibility(
                self.mrmlHierarchy.GetItemChildWithName(dissectionFolderId, DisplayMode.Body.name)
            )
            self.setVisibility(
                self.mrmlHierarchy.GetItemChildWithName(dissectionFolderId,
                DisplayMode.Slice.name),
                isVisible=False,
            )
        elif newDisplayMode == DisplayMode.Slice:
            self.setVisibility(
                self.mrmlHierarchy.GetItemChildWithName(dissectionFolderId,
                DisplayMode.Vertebra.name),
                isWireframe=True,
                opacity=0.2,
            )
            self.setVisibility(
                self.mrmlHierarchy.GetItemChildWithName(dissectionFolderId,
                DisplayMode.Body.name),
                opacity=0.2,
            )
            self.setVisibility(
                self.mrmlHierarchy.GetItemChildWithName(dissectionFolderId,
                DisplayMode.Slice.name),
                isWireframe=True,
                line_width=3.0,
            )
                
    def setVisibility(self, folder, *, isWireframe=False, isVisible=True, opacity=1.0, line_width=1.0):
        for childId in self.logic.getChildren(folder):
            node = self.mrmlHierarchy.GetDisplayNodeForItem(childId)
            if isVisible:
                node.VisibilityOn()
            else:
                node.VisibilityOff()
                continue

            if isWireframe:
                node.SetRepresentation(node.WireframeRepresentation)
                node.SetLineWidth(line_width)
            else:
                node.SetRepresentation(node.SurfaceRepresentation)

            node.SetOpacity(opacity)

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        inputGeometriesParameter = self._parameterNode.GetParameter("InputGeometries")
        displayModeParameter = self._parameterNode.GetParameter("DisplayMode")
        if inputGeometriesParameter and displayModeParameter:
            folderName = self.mrmlHierarchy.GetItemName(int(inputGeometriesParameter))
            self.ui.inputSelector.currentIndex = self.ui.inputSelector.findText(folderName)
            self.ui.displayModeComboBox.currentIndex = int(self._parameterNode.GetParameter("DisplayMode"))

        # Update buttons states and tooltips
        if self._parameterNode.GetParameter("InputGeometries"):
            self.ui.applyButton.toolTip = "Compute slope for geometry."
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input geometry."
            self.ui.applyButton.enabled = False
            self.ui.displayModeComboBox.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetParameter("InputGeometries", str(self.mrmlHierarchy.GetItemByName(self.ui.inputSelector.currentText)))
        self._parameterNode.SetParameter("DisplayMode", str(self.ui.displayModeComboBox.currentIndex))
        self._parameterNode.SetParameter("TopVertebra", str(self.ui.resultVertebraSelector.currentIndex))

        self._parameterNode.EndModify(wasModified)

    def initializeDisplayModeComboBox(self):
        self.ui.displayModeComboBox.addItems(DisplayMode.all_modes)
        self.ui.displayModeComboBox.currentIndex = 0

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        self.clearUI()
        
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            r = self.ui.rightVectorRSlider.value
            a = self.ui.rightVectorASlider.value
            s = self.ui.rightVectorSSlider.value
            rightVector = r, a, s
            self.logic.process(self.ui.inputSelector.currentText, rightVector)

            # Enable display mode changes
            self.ui.displayModeComboBox.enabled = True
            self.updateDisplayMode(DisplayMode.Vertebra.value)
            self.updateResultTable(self.logic.dimensions, names=self.logic.names)

            self.ui.resultsCollapsibleButton.collapsed = False
            self.ui.resultsCollapsibleButton.enabled = True

    def updateResultTable(self, dimensions: List[DimensionsLogic.Dimension], names: List[str]) -> None:
        self.ui.resultTableWidget.setRowCount(len(dimensions))
        self.ui.resultTableWidget.setColumnCount(10)
        self.ui.resultTableWidget.setHorizontalHeaderLabels([
            "Vertebra",
            "Upper Width",
            "Upper Depth",
            "Lower Width",
            "Lower Depth",
            "Height",
            "H (ant, sag)",
            "H (post, sag)",
            "H (ant, lat)",
            "H (post, lat)",
        ])

        for row, name, dimension in zip(count(), names, dimensions):
           nameItem = QTableWidgetItem(name) 
           widthItem = tuple(
                QTableWidgetItem(f"{dimension.width[endplate]:.3f}")
                for endplate in Endplate.options()
           )
           depthItem = tuple(
                QTableWidgetItem(f"{dimension.depth[endplate]:.3f}")
                for endplate in Endplate.options()
           )
           heightItem = QTableWidgetItem(f"{dimension.height[0]:.3f}")
           heightASItem = QTableWidgetItem(f"{dimension.height[1].far:.3f}")
           heightPSItem = QTableWidgetItem(f"{dimension.height[1].near:.3f}")
           heightALItem = QTableWidgetItem(f"{dimension.height[2].far:.3f}")
           heightPLItem = QTableWidgetItem(f"{dimension.height[2].near:.3f}")

           self.ui.resultTableWidget.setItem(row, 0, nameItem)
           self.ui.resultTableWidget.setItem(row, 1, widthItem[Endplate.UPPER])
           self.ui.resultTableWidget.setItem(row, 2, depthItem[Endplate.UPPER])
           self.ui.resultTableWidget.setItem(row, 3, widthItem[Endplate.LOWER])
           self.ui.resultTableWidget.setItem(row, 4, depthItem[Endplate.LOWER])
           self.ui.resultTableWidget.setItem(row, 5, heightItem)
           self.ui.resultTableWidget.setItem(row, 6, heightASItem)
           self.ui.resultTableWidget.setItem(row, 7, heightPSItem)
           self.ui.resultTableWidget.setItem(row, 8, heightALItem)
           self.ui.resultTableWidget.setItem(row, 9, heightPLItem)

    def enableSaveResultButton(self):
        self.ui.saveResultButton.enabled = True

    def onSaveResults(self):
        topmost_vertebra = Spine.VERTEBRAE.index(self.ui.resultVertebraSelector.currentText)
        self.logic.spine.name_vertebrae(offset_to_c1=topmost_vertebra)

        path = self.ui.resultPathLineEdit.currentPath
        Spine.write(self.logic.spine, path)
        logging.info(f"Inter-vertebral angles written to '{path}'")

    def selectedFolderId(self):
        return self.mrmlHierarchy.GetItemByName(self.ui.inputSelector.currentText)

    @property
    def selectedFolderName(self):
        return self.ui.inputSelector.currentText

    class DirectionVector:

        SelectorPostfix = "VectorSelector"
        RSliderPostfix = "VectorRSlider"
        ASliderPostfix = "VectorASlider"
        SSliderPostfix = "VectorSSlider"
        InvertButtonPostfix = "VectorInvertButton"

        def __init__(self, name, parent):
            self.parent = parent
            self.name = name
            self.trackedMarkupsLine = None

            uiElements = self.parent.ui.__dict__

            try:
                self.selector = uiElements[name + self.SelectorPostfix]
                self.rSlider = uiElements[name + self.RSliderPostfix]
                self.aSlider = uiElements[name + self.ASliderPostfix]
                self.sSlider = uiElements[name + self.SSliderPostfix]
                self.invertButton = uiElements[name + self.InvertButtonPostfix]
            except KeyError as err:
                from sys import stderr
                print(f"UI setup broken?\n{err} does not exist in UI file.", file=stderr, flush=True)
                return

            self.mrmlHierarchy = slicer.mrmlScene.GetSubjectHierarchyNode()
            self.updateMarkupsLinesSelector()
            if self.markupsLines:
                self.trackMarkupsLine(self.markupsLines[0])

            self.parent.addObserver(self.mrmlHierarchy, self.mrmlHierarchy.SubjectHierarchyItemAddedEvent, self.onHierarchyNodeAdded)
            self.selector.connect("currentIndexChanged(int)", self.onLineSelectorChanged)
            self.invertButton.connect("clicked(bool)", self.onInvertPressed)
            self.rSlider.connect("valueChanged(double)", self.onSliderChanged)
            self.aSlider.connect("valueChanged(double)", self.onSliderChanged)
            self.sSlider.connect("valueChanged(double)", self.onSliderChanged)

        def onHierarchyNodeAdded(self, node, event):
            newMarkupsLines = self.parent.logic.getUserMarkupsLines()
            if len(newMarkupsLines) == len(self.markupsLines):
                return

            newMarkupsLine = newMarkupsLines[-1]
            self.parent.addObserver(newMarkupsLine, newMarkupsLine.PointPositionDefinedEvent, self.onMarkupsLinePointAdded)

        def onMarkupsLinePointAdded(self, node, event):
            controlPointCount = node.GetNumberOfControlPoints()
            if controlPointCount < 2:
                return
            isFirstMarkupsLine = self.selector.currentIndex == -1

            self.markupsLines.append(node)
            self.selector.addItem(node.GetName())

            if not self.trackedMarkupsLine:
                self.adjustSliderToMarkupsLine(node)
                self.trackMarkupsLine(node)

        def onLineSelectorChanged(self, index):
            newMarkupsLine = self.markupsLines[self.selector.currentIndex]
            self.trackMarkupsLine(newMarkupsLine)

        def updateMarkupsLinesSelector(self):
            self.markupsLines = self.parent.logic.getUserMarkupsLines()
            self.selector.clear()
            self.selector.addItems([line.GetName() for line in self.markupsLines])

        def trackMarkupsLine(self, node):
            if self.trackedMarkupsLine and self.parent.hasObserver(
                    self.trackedMarkupsLine,
                    self.trackedMarkupsLine.PointModifiedEvent,
                    self.onMarkupsLineModified
            ):
                self.parent.removeObserver(self.trackedMarkupsLine, self.trackedMarkupsLine.PointModifiedEvent, self.onMarkupsLineModified)

            self.trackedMarkupsLine = slicer.mrmlScene.GetFirstNodeByName(self.selector.currentText)
            self.parent.addObserver(self.trackedMarkupsLine, node.PointModifiedEvent, self.onMarkupsLineModified)
            self.adjustSliderToMarkupsLine(self.trackedMarkupsLine)
            self.selector.enabled = True

        def onSliderChanged(self, newValue):
            self.adjustMarkupsLineToValues(
                    self.rSlider.value,
                    self.aSlider.value,
                    self.sSlider.value,
            )
            
        def onMarkupsLineModified(self, node, event):
            self.adjustSliderToMarkupsLine(node)

        def onInvertPressed(self, pressed):
            self.rSlider.value *= -1
            self.aSlider.value *= -1
            self.sSlider.value *= -1

        def adjustSliderToMarkupsLine(self, node):
            r, a, s = self.parent.logic.rasFromMarkupsLine(node)
            self.rSlider.value = r
            self.aSlider.value = a
            self.sSlider.value = s

        def adjustMarkupsLineToValues(self, *newValues):
            markupsLineIndex = self.selector.currentIndex
            markupsLine = self.markupsLines[markupsLineIndex]
            first = np.array(markupsLine.GetNthControlPointPosition(0))
            newDistance = np.array(newValues)
            newSecond = first + newDistance

            markupsLine.SetNthControlPointPosition(1, tuple(newSecond))


#
# DimensionsLogic
#

class DimensionsLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
    GeneratedAttribName = "generated"
    GeneratedWidthDirectory = "LowerWidth", "UpperWidth"
    GeneratedDepthDirectory = "LowerDepth", "UpperDepth"
    GeneratedHeightDirectory = "Height"
    GeneratedDimensionDirectories = {
        GeneratedWidthDirectory[Endplate.LOWER],
        GeneratedWidthDirectory[Endplate.UPPER],
        GeneratedDepthDirectory[Endplate.LOWER],
        GeneratedDepthDirectory[Endplate.UPPER],
        GeneratedHeightDirectory,
    }

    @dataclass
    class Dimension:
        width: Tuple[float, float]
        depth: Tuple[float, float]
        height: Tuple[float, float, float]

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.dimensions = [
            self.Dimension((0.0, 0.0,), (0.0, 0.0,), (0.0, 0.0, 0.0,))
            for _ in range(len(Spine.VERTEBRAE))
        ]
        self.names = copy(Spine.VERTEBRAE)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("DisplayMode"):
            parameterNode.SetParameter("DisplayMode", str(DisplayMode.Vertebra.value))

    def process(self, folderName, rasRightDirection, showResult=True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputGeometry: vertebra geometry to analyze
        :param showResult: show output volume in slice viewers
        """

        if not folderName:
            raise ValueError("Input geometry is invalid")

        mrmlHierarchy = slicer.mrmlScene.GetSubjectHierarchyNode()
        folderId = mrmlHierarchy.GetItemByName(folderName)
        geometries = [mrmlHierarchy.GetItemDataNode(child) for child in self.getChildren(folderId)]

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        self.run(geometries, rasRightDirection, folderName)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

    def run(self, geometries, rasRightDirection, folderName):
        mrmlScene = slicer.mrmlScene
        mrmlHierarchy: vtk.vtmMRMLSubjectHierarchyNode = mrmlScene.GetSubjectHierarchyNode()
        lpsRightDirection = np.array([1, -1, 1]) * rasRightDirection

        polydatas = [g.GetPolyData() for g in geometries]
        self.spine = Spine(polydatas, lateral_axis=np.array(lpsRightDirection), slice_thickness=0.25, max_angle=45.0)

        dissectionDirectory = mrmlHierarchy.CreateFolderItem(mrmlHierarchy.GetSceneItemID(), self.dissectionFolderName(folderName))
        vertebraDirectory = mrmlHierarchy.CreateFolderItem(dissectionDirectory, DisplayMode.Vertebra.name)
        bodyDirectory = mrmlHierarchy.CreateFolderItem(dissectionDirectory, DisplayMode.Body.name)
        sliceDirectory = mrmlHierarchy.CreateFolderItem(dissectionDirectory, DisplayMode.Slice.name)
        heightDirectory = mrmlHierarchy.CreateFolderItem(dissectionDirectory, self.GeneratedHeightDirectory)
        widthDirectory = tuple(
            mrmlHierarchy.CreateFolderItem(dissectionDirectory, self.GeneratedWidthDirectory[endplate])
            for endplate in Endplate.options()
        )
        depthDirectory = tuple(
            mrmlHierarchy.CreateFolderItem(dissectionDirectory, self.GeneratedDepthDirectory[endplate])
            for endplate in Endplate.options()
        )

        self.dimensions = []
        self.names = []
        for inputGeometry, vertebra in zip(geometries, self.spine):
            geometryName = inputGeometry.GetName()
            self.names.append(geometryName)

            # TODO: make this a dictionary and return to self.process
            
            width = []
            depth = []
            lateral_extrema = vertebra.body_laterally.minmax
            sagittal_extrema = vertebra.body.minmax
            for endplate in Endplate.options():
                firstPoint, lastPoint = lateral_extrema[endplate]
                width.append(np.linalg.norm(np.subtract(firstPoint, lastPoint)))
                self.addLine(
                    firstPoint,
                    lastPoint,
                    parentId=widthDirectory[endplate],
                    nodeName=geometryName + " - width",
                )
                firstPoint, lastPoint = sagittal_extrema[endplate]
                depth.append(np.linalg.norm(np.subtract(firstPoint, lastPoint)))
                self.addLine(
                    firstPoint,
                    lastPoint,
                    parentId=depthDirectory[endplate],
                    nodeName=geometryName + " - depth",
                )

            self.addLine(vertebra.center[Endplate.LOWER], vertebra.center[Endplate.UPPER], parentId=heightDirectory, nodeName=geometryName + " - height")
            sagittal_rim_heights = vertebra.body.height
            lateral_rim_heights = vertebra.body_laterally.height
            height = (
                np.linalg.norm(np.subtract(vertebra.center[Endplate.UPPER], vertebra.center[Endplate.LOWER])),
                vertebra.body.height,
                vertebra.body_laterally.height,
            )
            self.add(vertebra.geometry, parentId=vertebraDirectory, name=geometryName)
            self.add(vertebra.body.endplates, parentId=bodyDirectory, name=geometryName)
            self.add(vertebra.body.curves[Endplate.UPPER], parentId=sliceDirectory, name=geometryName)

            self.dimensions.append(self.Dimension(width, depth, height))
                
    def add(self, polydata, parentId, name):
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        modelNode.SetAndObservePolyData(polydata)
        modelNode.CreateDefaultDisplayNodes()
        #print(f"{name}, generated: {modelNode.GetAttribute(self.GeneratedAttribName)}")

        mrmlHierarchy = slicer.mrmlScene.GetSubjectHierarchyNode()
        modelNodeId = mrmlHierarchy.GetItemByDataNode(modelNode)
        mrmlHierarchy.SetItemParent(modelNodeId, parentId)
        mrmlHierarchy.SetItemAttribute(modelNodeId, self.GeneratedAttribName, str(True))

    def addLine(self, firstPoint, secondPoint, parentId, nodeName):
        markupsLine = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", nodeName)
        markupsLine.AddControlPoint(firstPoint)
        markupsLine.AddControlPoint(secondPoint)
        #print(f"{nodeName}, generated: {markupsLine.GetAttribute(self.GeneratedAttribName)}")

        mrmlHierarchy = slicer.mrmlScene.GetSubjectHierarchyNode()
        markupsLineId = mrmlHierarchy.GetItemByDataNode(markupsLine)
        mrmlHierarchy.SetItemParent(markupsLineId, parentId)
        mrmlHierarchy.SetItemAttribute(markupsLineId, self.GeneratedAttribName, str(True))

    @classmethod
    def dissectionFolderContent(cls, name):
        mrmlHierarchy = slicer.mrmlScene.GetSubjectHierarchyNode()

        dissectionFolderName = cls.dissectionFolderName(name)
        dissectionFolderId = mrmlHierarchy.GetItemByName(dissectionFolderName)
        dissectionIds = list()
        mrmlHierarchy.GetItemChildren(dissectionFolderId, dissectionIds)

        return [mrmlHierarchy.GetItemDataNode(id_) for id_ in dissectionIds]

    @staticmethod
    def dissectionFolderName(name):
        return f"{name} dissection"

    @classmethod
    def dissectionFolderId(cls, name):
        mrmlHierarchy = slicer.mrmlScene.GetSubjectHierarchyNode()
        return mrmlHierarchy.GetItemByName(cls.dissectionFolderName(name))

    @staticmethod
    def getParentName(node):
        mrmlHierarchy = slicer.mrmlScene.GetSubjectHierarchyNode()
        nodeId = mrmlHierarchy.GetItemByDataNode(node)
        parentId = mrmlHierarchy.GetItemParent(nodeId)
        return mrmlHierarchy.GetItemName(parentId)

    @classmethod
    def getUserMarkupsLines(cls):
        return [l for l in slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsLineNode") if cls.getParentName(l) not in cls.GeneratedDimensionDirectories]

    @staticmethod
    def rasFromMarkupsLine(markupsLine):
        first = np.array(markupsLine.GetNthControlPointPosition(0))
        second = np.array(markupsLine.GetNthControlPointPosition(1))
        
        return tuple(second - first)


    def extractBody(self, polydata):
        copy = vtk.vtkPolyData()
        copy.DeepCopy(polydata)
        return copy

    def extractSlice(self, polydata):
        return self.extractBody(polydata)

    def extractSlope(self, polydata):
        return self.extractBody(polydata)

    @classmethod
    def isItemAFolder(cls, item_id: int):
        mrmlHierarchy = slicer.mrmlScene.GetSubjectHierarchyNode()
        return mrmlHierarchy.GetNumberOfItemChildren(item_id) > 0

    @staticmethod
    def getChildren(item_id: int):
        children = list()
        mrmlHierarchy = slicer.mrmlScene.GetSubjectHierarchyNode()
        mrmlHierarchy.GetItemChildren(item_id, children)
        return children


#
# DimensionsTest
#

class DimensionsTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_Dimensions1()

    def test_Dimensions1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('Dimensions1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = DimensionsLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
