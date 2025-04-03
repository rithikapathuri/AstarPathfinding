import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
import imageio
import cv2
import heapq
import math

class cell:
    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0
        self.parent = None
        self.blocked = False
        
    def __lt__(self, other):
        return self.f < other.f

def heuristic(start, end):
    maximum = max(abs(start.x - end.x), abs(start.z - end.z))
    minimum = min(abs(start.x - end.x), abs(start.z - end.z))
    x = (maximum - minimum) + (minimum * math.sqrt(2))
    return x
    
def getNeighborCells(grid, node, offsets, speed, factor, gaussianMap):
    neighbors = []
    for dx, dz, cost in offsets:
        nx, nz = node.x + dx, node.z + dz
        if 0 <= nx < grid.shape[0] and 0 <= nz < grid.shape[1]: 
            neighbor = grid[nx, nz]
            if not neighbor.blocked:
                influenceCost = factor * (gaussianMap[node.z, node.x] + gaussianMap[nz, nx]) / 2
                print(f"distance cost {cost} and influence cost {influenceCost}")
                neighbors.append((neighbor, influenceCost))
    return neighbors

def aStar(binaryMap, grid, start, goal, offsets, speed, factor, gaussianMap, obstacles):
    openList = []
    closedList = set()
    startCell = grid[start[0], start[1]]
    goalCell = grid[goal[0], goal[1]]
    startCell.g = 0
    startCell.h = heuristic(startCell, goalCell)
    startCell.h = 0
    startCell.f = startCell.g + startCell.h
    heapq.heappush(openList, (startCell.f, startCell))
    while openList:
        _, current = heapq.heappop(openList)
        if current in closedList:
            continue
        if current.x == goalCell.x and current.z == goalCell.z:
            timeInterval = 100
            path = []
            while current:
                path.append((current.x, current.z, timeInterval))
                current = current.parent
                timeInterval -= 0.3
            path = path[::-1]
            print(path)
            return path
        closedList.add(current)
        neighbors = getNeighborCells(grid, current, offsets, speed, factor, gaussianMap)
        for neighbor, cost in neighbors:
            if neighbor in closedList:
                continue
            temp_g = current.g + cost
            if temp_g < neighbor.g:
                neighbor.g = temp_g
                #neighbor.h = heuristic(neighbor, goalCell)
                neighbor.h = 0
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current
                heapq.heappush(openList, (neighbor.f, neighbor))
    
    return None

def plotXZOverTime(path, interval, gridSize, staticObstacles, movingObstacles):
    numFrames = 10
    filename = 'xz_time_progression2.gif'
    
    fig, ax = plt.subplots(figsize=(6, 6))
    xs, zs, ys = zip(*path)
    print(ys)
    minTime, maxTime = min(ys), max(ys)
    print(minTime)
    print(maxTime)
    intervals = np.linspace(minTime, maxTime, numFrames)
    images = []
    
    
    for t in intervals:
        ax.clear()
        ax.set_xlim(0, gridSize[0])
        ax.set_ylim(0, gridSize[1])
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Z Axid')
        ax.set_title(f'XZ View at Time {round(t - minTime, 2)}')
        
        plt.plot(start[0], start[1], 'bo', label = "start")  #plot start point -> blue
        plt.plot(goal[0], goal[1], 'go', label = "goal")    #plot end point -> green
                
        for obs in staticObstacles:
            ax.scatter(*obs, c='black', marker='s', label='Static' if obs == list(staticObstacles)[0] else "")
        
        points = [(x, z) for x, z, y in zip(xs, zs, ys) if y <= t]
        if points:
            xVals, zVals = zip(*points)
            ax.scatter(xVals, zVals, c='blue', marker='o')
        
        plt.draw()
        plt.pause(0.001)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image = np.array(fig.canvas.buffer_rgba()) 
        expectedSize = (height, width, 4)
        if image.shape != expectedSize:
            image = cv2.resize(image, (width, height))
        image = image[:, :, :3]
        images.append(image)
    
        imageio.mimsave(filename, images, fps=2)
    plt.show()

def createGradientMap(binaryMap, start, goal, obstacles, offsets):
    gradientMap = np.full(binaryMap.shape, np.inf, dtype=float)
    pq = []
    for x, z in obstacles:
        gradientMap[z, x] = 0
        heapq.heappush(pq, (0, (x, z)))
    while pq:
        cost, (x, z) = heapq.heappop(pq)
        for dx, dz, moveCost in offsets:
            nx, nz = x + dx, z + dz
            if 0 <= nx < binaryMap.shape[0] and 0 <= nz < binaryMap.shape[1]:
                newCost = cost + moveCost
                if newCost < gradientMap[nz, nx]:
                    gradientMap[nz, nx] = newCost
                    heapq.heappush(pq, (newCost, (nx, nz)))
    maxVal = np.max(gradientMap[gradientMap != np.inf])
    gradientMap = gradientMap / maxVal
    #print("MAXVAL: "+str(maxVal))
    return gradientMap

def plotGradientMap(gradientMap, start, goal, width, length):
    norm = Normalize(vmin=0, vmax=1)
    plt.figure(figsize=(8, 8))
    colors = {
        'red':   ((0.0, 1.0, 1.0),
                (0.17, 1.0, 1.0),
                (0.33, 1.0, 1.0),
                (0.5, 0.0, 0.0),
                (0.67, 0.0, 0.0),
                (0.83, 0.5, 0.5),
                (1.0, 0.5, 0.5)),
        'green': ((0.0, 0.0, 0.0),
                (0.17, 0.5, 0.5),
                (0.33, 1.0, 1.0),
                (0.5, 1.0, 1.0),
                (0.67, 1.0, 1.0),
                (0.83, 0.0, 0.0),
                (1.0, 0.0, 0.0)),
        'blue':  ((0.0, 0.0, 0.0),
                (0.17, 0.0, 0.0),
                (0.33, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (0.67, 1.0, 1.0),
                (0.83, 1.0, 1.0),
                (1.0, 0.5, 0.5))
    }
    cmap = LinearSegmentedColormap('my_colormap',colors,256)   
    plt.imshow(gradientMap, cmap=cmap, norm=norm, origin='lower', interpolation='nearest', extent=(0, width, 0, length)) #plot gradient map as heatmap
    plt.plot(start[0], start[1], 'bo', label = "start")  #plot start point -> blue
    plt.plot(goal[0], goal[1], 'go', label = "goal")    #plot end point -> green
    plt.title("Environment Gradient Map")
    plt.xlabel("X Axis")
    plt.ylabel("Z Axis")
    plt.legend()
    plt.show()

def createGaussianMap(gradientMap, width, length):
    gaussianMap = np.zeros((width, length))
    X, Z = np.meshgrid(np.arange(width), np.arange(length))
    #temp = float(sum([gradientMap[z, x] for x in range(width) for z in range(length)])) / (width*length)
    variances = [0.1, 0.25, 0.75]
    var = variances[2]
    mean = 0
    #print(mean)
    for x in range(width):
        for z in range(length):
            distance = gradientMap[z, x]
            influence = 1 / (math.sqrt(var * 2 * math.pi))
            gaussian = influence * np.exp(-(((distance - mean) ** 2) / (2 * var)))
            gaussianMap[z, x] = gaussian
    maxVal = np.max(gaussianMap[gaussianMap != np.inf])
    gaussianMap = gaussianMap / maxVal
    #print("MAXVAL: "+str(maxVal))
    return gaussianMap

def plotGaussianMap(gaussianMap, start, goal, width, length):
    norm = Normalize(vmin=0, vmax=1)
    plt.figure(figsize=(8, 8))
    colors = {
        'red':   ((0.0, 1.0, 1.0),
                (0.17, 1.0, 1.0),
                (0.33, 1.0, 1.0),
                (0.5, 0.0, 0.0),
                (0.67, 0.0, 0.0),
                (0.83, 0.5, 0.5),
                (1.0, 0.5, 0.5)),
        'green': ((0.0, 0.0, 0.0),
                (0.17, 0.5, 0.5),
                (0.33, 1.0, 1.0),
                (0.5, 1.0, 1.0),
                (0.67, 1.0, 1.0),
                (0.83, 0.0, 0.0),
                (1.0, 0.0, 0.0)),
        'blue':  ((0.0, 0.0, 0.0),
                (0.17, 0.0, 0.0),
                (0.33, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (0.67, 1.0, 1.0),
                (0.83, 1.0, 1.0),
                (1.0, 0.5, 0.5))
    }
    cmap = LinearSegmentedColormap('my_colormap',colors,256)   
    plt.imshow(gaussianMap, cmap=cmap, norm=norm, origin='lower', interpolation='nearest', extent=(0, width, 0, length)) #plot gradient map as heatmap
    plt.plot(start[0], start[1], 'bo', label = "start")  #plot start point -> blue
    plt.plot(goal[0], goal[1], 'go', label = "goal")    #plot end point -> green
    plt.title("Environment Gaussian Map")
    plt.xlabel("X Axis")
    plt.ylabel("Z Axis")
    plt.legend()
    plt.show()

    

offsets = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2))]
width, length = 100, 100
grid = np.array([[cell(x, z) for z in range(width)] for x in range(length)])
binaryMap = np.zeros((width, length))
gradientMap = np.zeros((width, length))
time_step = 0.05
speed = 3
benches = []
walls = []

#benches in center
for x in range(40, 60):
    for z in range(35, 65):
        benches.append((x, z))
        binaryMap[z, x] = 1
        grid[x, z].blocked = True

#bottom wall
for x in range(0, 100):
    for z in range(0, 2):
        walls.append((x, z))
        binaryMap[z, x] = 1
        grid[x, z].blocked = True

#top wall
for x in range(0, 100):
    for z in range(98, 100):
        if not(x > 15 and x < 35) and not(x > 65 and x < 85):
            walls.append((x, z))
            binaryMap[z, x] = 1
            grid[x, z].blocked = True

#left wall
for x in range(0, 2):
    for z in range(0, 100):
        if not(z > 15 and z < 35) and not(z > 65 and z < 85):
            walls.append((x, z))
            binaryMap[z, x] = 1
            grid[x, z].blocked = True

#right wall
for x in range(98, 100):
    for z in range(0, 100):
        if not(z > 15 and z < 35) and not(z > 65 and z < 85):
            walls.append((x, z))
            binaryMap[z, x] = 1
            grid[x, z].blocked = True

#start = (50, 5)
start = (99, 80)
#goal = (75, 99) #right side goal door
goal = (99, 20)
#gradientMapBenches = createGradientMap(binaryMap, start, goal, benches, offsets)
#gradientMapWalls = createGradientMap(binaryMap, start, goal, walls, offsets)
gradientMapObstacles = createGradientMap(binaryMap, start, goal, benches+walls, offsets)
#plotGradientMap(gradientMapObstacles, start, goal, width, length)  #plotting gradient map
gaussianMapObstacles = createGaussianMap(gradientMapObstacles, width, length)
#plotGaussianMap(gaussianMapObstacles, start, goal, width, length)  #plotting gradient map

factor = 1.251
staticObstacles = benches + walls
movingObstacles = set()
movingObstacles.add((40, 19))
path = None
path = aStar(binaryMap, grid, start, goal, offsets, speed, factor, gaussianMapObstacles, staticObstacles)

if path:
    plotXZOverTime(path, 1, (length, width), staticObstacles, movingObstacles)
else:
    print("No path found.")