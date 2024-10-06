import numpy as np
from scipy import ndimage
import imageio
from PIL import Image, ImageFilter
import constants
import imageio.v2 as imageio

class CircularRange:
    def __init__(self, begin, end, value):
        self.begin, self.end, self.value = begin, end, value

    def __repr__(self):
        return f"[{self.begin},{self.end})->{self.value}"

    def halfway(self):
        return int((self.begin + self.end) / 2)

class Graph:
    class Node:
        def __init__(self, point, index):
            self.x, self.y = point
            self.index = index
            self.connections = {}

        def __repr__(self):
            return f"({self.y},{-self.x})"

        def _addConnection(self, to):
            self.connections[to] = False # i.e. not already used in gcode generation

        def toDotFormat(self):
            return (f"{self.index} [pos=\"{self.y},{-self.x}!\", label=\"{self.index}\\n{self.x},{self.y}\"]\n" +
                "".join(f"{self.index}--{conn}\n" for conn in self.connections if self.index < conn))

    def __init__(self):
        self.nodes = []

    def __getitem__(self, index):
        return self.nodes[index]

    def __repr__(self):
        return repr(self.nodes)

    def addNode(self, point):
        index = len(self.nodes)
        self.nodes.append(Graph.Node(point, index))
        return index

    def addConnection(self, a, b):
        self.nodes[a]._addConnection(b)
        self.nodes[b]._addConnection(a)

    def distance(self, a, b):
        return np.hypot(self[a].x-self[b].x, self[a].y-self[b].y)

    def areConnectedWithin(self, a, b, maxDistance):
        if maxDistance < 0:
            return False
        elif a == b:
            return True
        else:
            for conn in self[a].connections:
                if self.areConnectedWithin(conn, b, maxDistance - self.distance(conn, b)):
                    return True
            return False

    def saveAsDotFile(self, f):
        f.write("graph G {\nnode [shape=plaintext];\n")
        for node in self.nodes:
            f.write(node.toDotFormat())
        f.write("}\n")

    def saveAsGcodeFile(self, f):
        def pathGcode(i, insidePath):
            f.write(f"G{1 if insidePath else 0} X{self[i].y} Y{-self[i].x}\n")
            for connTo, alreadyUsed in self[i].connections.items():
                if not alreadyUsed:
                    self[i].connections[connTo] = True
                    self[connTo].connections[i] = True
                    return pathGcode(connTo, True)
            return i

        possibleStartingNodes = set()
        for i in range(len(self.nodes)):
            if len(self[i].connections) == 0 or len(self[i].connections) % 2 == 1:
                possibleStartingNodes.add(i)

        if len(possibleStartingNodes) != 0:
            node = next(iter(possibleStartingNodes)) # first element
            while 1:
                possibleStartingNodes.remove(node)
                pathEndNode = pathGcode(node, False)

                if len(self[node].connections) == 0:
                    assert pathEndNode == node
                    f.write(f"G1 X{self[node].y} Y{-self[node].x}\n")
                else:
                    possibleStartingNodes.remove(pathEndNode)

                if len(possibleStartingNodes) == 0:
                    break

                minDistanceSoFar = np.inf
                for nextNode in possibleStartingNodes:
                    distance = self.distance(pathEndNode, nextNode)
                    if distance < minDistanceSoFar:
                        minDistanceSoFar = distance
                        node = nextNode

        cycleNodes = set()
        for i in range(len(self.nodes)):
            someConnectionsAvailable = False
            for _, alreadyUsed in self[i].connections.items():
                if not alreadyUsed:
                    someConnectionsAvailable = True
                    break

            if someConnectionsAvailable:
                cycleNodes.add(i)

        def cyclePathGcode(i, insidePath):
            f.write(f"G{1 if insidePath else 0} X{self[i].y} Y{-self[i].x}\n")

            foundConnections = 0
            for connTo, alreadyUsed in self[i].connections.items():
                if not alreadyUsed:
                    if foundConnections == 0:
                        self[i].connections[connTo] = True
                        self[connTo].connections[i] = True
                        cyclePathGcode(connTo, True)

                    foundConnections += 1
                    if foundConnections > 1:
                        break

            if foundConnections == 1:
                cycleNodes.remove(i)

        if len(cycleNodes) != 0:
            node = next(iter(cycleNodes)) # first element
            while 1:
                cyclePathGcode(node, False)

                if len(cycleNodes) == 0:
                    break

                pathEndNode = node
                minDistanceSoFar = np.inf
                for nextNode in possibleStartingNodes:
                    distance = self.distance(pathEndNode, nextNode)
                    if distance < minDistanceSoFar:
                        minDistanceSoFar = distance
                        node = nextNode

class EdgesToGcode:
    def __init__(self, edges):
        self.edges = edges
        self.ownerNode = np.full(np.shape(edges), -1, dtype=int)
        self.xSize, self.ySize = np.shape(edges)
        self.graph = Graph()

    def getCircularArray(self, center, r, smallerArray = None):
        circumferenceSize = len(constants.circumferences[r])
        circularArray = np.zeros(circumferenceSize, dtype=bool)

        if smallerArray is None:
            smallerArray = np.ones(1, dtype=bool)
        smallerSize = np.shape(smallerArray)[0]
        smallerToCurrentRatio = smallerSize / circumferenceSize

        for i in range(circumferenceSize):
            x = center[0] + constants.circumferences[r][i][0]
            y = center[1] + constants.circumferences[r][i][1]

            if x not in range(self.xSize) or y not in range(self.ySize):
                circularArray[i] = False # consider pixels outside of the image as not-edges
            else:
                iSmaller = i * smallerToCurrentRatio
                a, b = int(np.floor(iSmaller)), int(np.ceil(iSmaller))

                if smallerArray[a] == False and (b not in range(smallerSize) or smallerArray[b] == False):
                    circularArray[i] = False # do not take into consideration not connected regions (roughly)
                else:
                    circularArray[i] = self.edges[x, y]

        return circularArray

    def toCircularRanges(self, circularArray):
        ranges = []
        circumferenceSize = np.shape(circularArray)[0]

        lastValue, lastValueIndex = circularArray[0], 0
        for i in range(1, circumferenceSize):
            if circularArray[i] != lastValue:
                ranges.append(CircularRange(lastValueIndex, i, lastValue))
                lastValue, lastValueIndex = circularArray[i], i

        ranges.append(CircularRange(lastValueIndex, circumferenceSize, lastValue))
        if len(ranges) > 1 and ranges[-1].value == ranges[0].value:
            ranges[0].begin = ranges[-1].begin - circumferenceSize
            ranges.pop() # the last range is now contained in the first one
        return ranges

    def getNextPoints(self, point):
        bestRadius = 0
        circularArray = self.getCircularArray(point, 0)
        allRanges = [self.toCircularRanges(circularArray)]
        for radius in range(1, len(constants.circumferences)):
            circularArray = self.getCircularArray(point, radius, circularArray)
            allRanges.append(self.toCircularRanges(circularArray))
            if len(allRanges[radius]) > len(allRanges[bestRadius]):
                bestRadius = radius
            if len(allRanges[bestRadius]) >= 4 and len(allRanges[-2]) >= len(allRanges[-1]):
                break
            elif len(allRanges[radius]) == 2 and radius > 1:
                edge = 0 if allRanges[radius][0].value == True else 1
                if allRanges[radius][edge].end-allRanges[radius][edge].begin < len(constants.circumferences[radius]) / 4:
                    if bestRadius == 1:
                        bestRadius = 2
                    break
            elif len(allRanges[radius]) == 1 and allRanges[radius][0].value == False:
                break

        if bestRadius == 0:
            return 0, []

        circularRanges = allRanges[bestRadius]
        points = []
        for circularRange in circularRanges:
            if circularRange.value == True:
                circumferenceIndex = circularRange.halfway()
                x = point[0] + constants.circumferences[bestRadius][circumferenceIndex][0]
                y = point[1] + constants.circumferences[bestRadius][circumferenceIndex][1]

                if x in range(self.xSize) and y in range(self.ySize) and self.ownerNode[x, y] == -1:
                    points.append((x,y))

        return bestRadius, points

    def propagate(self, point, currentNodeIndex):
        radius, nextPoints = self.getNextPoints(point)

        allConnectedNodes = set()
        def setSeenDFS(x, y):
            if (x in range(self.xSize) and y in range(self.ySize)
                    and np.hypot(x-point[0], y-point[1]) <= radius + 0.5
                    and self.edges[x, y] == True and self.ownerNode[x, y] != currentNodeIndex):
                if self.ownerNode[x, y] != -1:
                    allConnectedNodes.add(self.ownerNode[x, y])
                self.ownerNode[x, y] = currentNodeIndex # index of just added node
                setSeenDFS(x+1, y)
                setSeenDFS(x-1, y)
                setSeenDFS(x, y+1)
                setSeenDFS(x, y-1)

        self.ownerNode[point] = -1 # reset to allow DFS to start
        setSeenDFS(*point)
        for nodeIndex in allConnectedNodes:
            if not self.graph.areConnectedWithin(currentNodeIndex, nodeIndex, 11):
                self.graph.addConnection(currentNodeIndex, nodeIndex)

        validNextPoints = []
        for nextPoint in nextPoints:
            if self.ownerNode[nextPoint] == currentNodeIndex:
                validNextPoints.append(nextPoint)

        for nextPoint in validNextPoints:
            nodeIndex = self.graph.addNode(nextPoint)
            self.graph.addConnection(currentNodeIndex, nodeIndex)
            self.propagate(nextPoint, nodeIndex)
            self.ownerNode[point] = currentNodeIndex

    def addNodeAndPropagate(self, point):
        nodeIndex = self.graph.addNode(point)
        self.propagate(point, nodeIndex)

    def buildGraph(self):
        for point in np.ndindex(np.shape(self.edges)):
            if self.edges[point] == True and self.ownerNode[point] == -1:
                radius, nextPoints = self.getNextPoints(point)
                if radius == 0:
                    self.addNodeAndPropagate(point)
                else:
                    for nextPoint in nextPoints:
                        if self.ownerNode[nextPoint] == -1:
                            self.addNodeAndPropagate(nextPoint)

        return self.graph

def sobel(image):
    image = np.array(image, dtype=float)
    image /= 255.0
    Gx = ndimage.sobel(image, axis=0)
    Gy = ndimage.sobel(image, axis=1)
    res = np.hypot(Gx, Gy)
    res /= np.max(res)
    res = np.array(res * 255, dtype=np.uint8)
    return res[2:-2, 2:-2, 0:3]

def convertToBinaryEdges(edges, threshold):
    result = np.maximum.reduce([edges[:, :, 0], edges[:, :, 1], edges[:, :, 2]]) >= threshold
    if np.shape(edges)[2] > 3:
        result[edges[:, :, 3] < threshold] = False
    return result

def main():
    # Hardcode the input and output file paths
    input_file = r"C:\Users\mjain\Downloads\image-to-gcode-master\Oblivion_gcode_generator\images\example3.jpg" 
    output_file = r"C:\Users\mjain\Downloads\image-to-gcode-master\Oblivion_gcode_generator\images\example3_gcode.jpg" 
    dot_output_file = None
    
    # Set other parameters
    edges_mode = None  # Set to "white" or "black" if needed, or None for default behavior
    threshold = 32  # You can change this value if needed

    try:
        # Read the input image
        image = imageio.imread(input_file)
        
        if edges_mode is None:
            edges = sobel(image)
        elif edges_mode == "black":
            edges = np.invert(image)
        else:  # edges_mode == "white"
            edges = image
        
        edges = convertToBinaryEdges(edges, threshold)

        converter = EdgesToGcode(edges)
        converter.buildGraph()

        if dot_output_file is not None:
            with open(dot_output_file, 'w') as dot_file:
                converter.graph.saveAsDotFile(dot_file)
        
        with open(output_file, 'w') as output:
            converter.graph.saveAsGcodeFile(output)
        
        print(f"Processing complete. Output saved to {output_file}")
        if dot_output_file:
            print(f"Dot file saved to {dot_output_file}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except IOError as e:
        print(f"Error: IO error occurred - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
