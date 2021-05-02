import numpy as np
import pandas as pd
import scipy.spatial.distance
import matplotlib.pyplot as plt  
import cartopy.crs as ccrs 
import itertools


n = 500        # number of airports
distExp = -2.08 # gravity law demand distance exponent
zipfExp = 1.07  # zipf's law population distribution exponent

doOpt = False
printUpdates = True
rng = np.random.default_rng()
def toDeg(angles):
    return [angles[0]*180/np.pi - 90, angles[1]*180/np.pi - 180]
def toLat(angles):
    return [(angles[0]+90)*np.pi/180, (angles[1]+180)*np.pi/180]
def dist(a, b):
    return np.arccos(np.cos(a[0]) * np.cos(b[0]) + np.sin(a[0]) * np.sin(b[0]) * np.cos(a[1] - b[1])-1e-8)
def demand(p1, p2, d):
    if d < .001:
        return 0
    return p1*p2*d**distExp
def sampleSphere(n):
    coords = np.zeros((n, 2))
    for i in range(n):
        coords[i][0] = np.arccos(rng.uniform(-1, 1)) # theta
        coords[i][1] = rng.uniform(0, 2*np.pi)       # phi
    return coords
def sampleLine(n):
    coords = np.zeros((n, 2))
    for i in range(n):
        coords[i][0] = np.pi/2
        coords[i][1] = rng.uniform(0, 2*np.pi)
    return coords
def sampleTwo(n):
    coords = np.zeros((n, 2))
    for i in range(n):
        coords[i][0] = rng.uniform(15*np.pi/32, 17*np.pi/32)
        if i < n//2:
            coords[i][1] = rng.uniform(0, np.pi/16)
        else:
            coords[i][1] = rng.uniform(13*np.pi/16., 14*np.pi/16)
    return coords
def cost(plane, distance):
    return ((plane**.9)*distance)

def generateCities(shape = 'random', popDist = 'zipf'):
    if shape == 'random':
        ports = np.array(sampleSphere(n))
    if shape == 'line':
        ports = np.array(sampleLine(n))
    if shape == 'two clusters':
        ports = np.array(sampleTwo(n))
    if popDist == 'zipf':
        populations = rng.zipf(zipfExp, n).astype(float)
    if popDist == 'uniform':
        populations = rng.uniform(1, 1000, n).astype(float)
    if popDist == 'constant':
        populations = np.ones(n)*10
    ports = [port for _, port in sorted(zip(populations, ports), key=lambda pair: pair[1][1])]
    populations = [pop for pop, _ in sorted(zip(populations, ports), key=lambda pair: pair[1][1])]
    distances = scipy.spatial.distance.cdist(ports, ports, metric=dist)
    ports = [toDeg(port) for port in ports]
    return (ports, populations, distances)
def getCities(distr = 'top'):
    df = pd.read_csv('./worldcities.csv', sep=',')
    if distr == 'top':
        df = df[:n]
    if distr == 'random':
        df = df[:-4][['lat', 'lng', 'population']].dropna().sample(n=n).sort_values('population')
    ports = np.array(df[['lat', 'lng']])
    populations = (np.array(df[['population']])).flatten()
    distances = scipy.spatial.distance.cdist([toLat(port) for port in ports], [toLat(port) for port in ports], metric=dist)
    return (ports, populations, distances)
def powerset(iterable):
    "list(powerset([1,2,3])) --> [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

#ports, populations, distances = generateCities('random', 'uniform')
ports, populations, distances = getCities()
demands = np.array([[demand(populations[i], populations[j], distances[i][j]) for i in range(n)] for j in range(n)], ndmin=2)
planes = np.copy(demands)


def symlog(x):
    if abs(x) < 1:
        return x
    if x > 1:
        return np.log(x)+1
    if x < 1:
        return -np.log(-x)-1

symlogV = np.vectorize(symlog)
def plot(planes, optPlanes = []):
    if len(optPlanes):
        fig = plt.figure()
        plt.subplots_adjust(wspace = 0.2, hspace = 0.5 )
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_title('Demand')
        ax1.matshow(np.log(demands))
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_title('Iterative Sol.')
        ax2.matshow(np.log(planes))
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title('Optimal Sol.')
        ax3.matshow(np.log(optPlanes))
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title('Iterative - Optimal')
        ax4.matshow(symlogV(planes-optPlanes))
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.matshow(np.log(demands))
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.matshow(np.log(planes))
    plt.show()

    fig = plt.figure()
    if len(optPlanes):
        ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
    else:
        ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    biggestShade = np.amax(np.log(planes/np.min(planes[np.nonzero(planes)])))
    for i in range(n):
        for j in range(i, n):
            if planes[i][j]:
                #ax1.plot([ports[i][1], ports[j][1]], [ports[i][0], ports[j][0]], transform=ccrs.Geodetic(), c='crimson', alpha=max(.1, min(1, np.log(planes[i][j])/(n))))
                ax1.plot([ports[i][1], ports[j][1]], [ports[i][0], ports[j][0]], transform=ccrs.Geodetic(), c='crimson', alpha=np.log(planes[i][j]/np.min(planes[np.nonzero(planes)]))/biggestShade)
    ax1.scatter([port[1] for port in ports], [port[0] for port in ports], c=np.log(populations)/2.302)
    if len(optPlanes):
        ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
        for i in range(n):
            for j in range(i, n):
                if planes[i][j]:
                    ax2.plot([ports[i][1], ports[j][1]], [ports[i][0], ports[j][0]], transform=ccrs.Geodetic(), c='crimson', alpha=max(.1, min(1, 2*np.log(optPlanes[i][j])/(n))))
        ax2.scatter([port[1] for port in ports], [port[0] for port in ports], c=np.log(populations)/2.302)
    plt.show()

counter = 0
globalBestScore = np.inf
pairsByDemand = sorted([(i%n, (i-i%n)//n, demands[i%n][(i-i%n)//n]) for i in range(n**2)], key = lambda a: -a[2])
pairsByDemand = [(a[0], a[1]) for a in pairsByDemand if (a[0] > a[1])]
def solveOptimalOuter():
    def solveOptimal(depth, p):
        global globalBestScore
        global counter
        counter += 1

        score = sum([sum([cost(p[a][b], distances[a][b]) for a in range(n)]) for b in range(n)])
        if (score > globalBestScore):
            return ([np.inf, []]) 
        if (depth == len(pairsByDemand)):
            if score < globalBestScore:
                print(score)
                globalBestScore = score 
                return ([score, p])
            else:
                return ([np.inf, []])
        
        (j, i) = pairsByDemand[depth]
        bestChildScore = (np.inf, [])
        bestRoute = []
        otherPorts = list(range(n)[:i]) + list(range(n)[i+1:j]) + list(range(n)[j+1:])
        routeSets = powerset(otherPorts)
        routes = []
        for routeSet in routeSets:
            routes += itertools.permutations(routeSet)
        for route in routes:
            routeContribution = np.zeros((n, n))
            if len(route) == 0:
                routeContribution[i][j] = demands[i][j]
            else:
                routeContribution[i][route[0]] = demands[i][j]
                routeContribution[route[-1]][j] = demands[i][j]
                for stop in range(len(route)-1):
                    routeContribution[route[stop]][route[stop+1]] = demands[i][j]
            pNew = p + routeContribution + np.transpose(routeContribution)
            childScore = []
            if j == n-1:
                childScore = solveOptimal(depth+1, pNew)
            else:
                childScore = solveOptimal(depth+1, pNew)
            
            if childScore[0] < bestChildScore[0]:
                bestRoute = route
                bestChildScore = childScore
        return bestChildScore

    planes = np.copy(demands)
    return (solveOptimal(0, np.zeros((n, n))))
if doOpt:
    (optScore, optPlanes) = solveOptimalOuter()
    print(optScore)
    print(counter)
            



attempts = 1
maxIter = 10000000000
smallUpdate = 10000
update = 10000
bestFinalCost = np.inf
bestPlanes = []
bestPorts = []
bestCostPlot = []
orderByDemand = False

for j in range(attempts):
    print('--------------------------')
    print('attempt ' + str(j))
    planes = np.copy(demands)
    costPlot = []
    distPlot = []
    changes = 0

    aList = np.arange(n)
    bList = np.arange(n)
    rng.shuffle(aList)
    rng.shuffle(bList)
    for i in range(1, maxIter):
        a = aList[i%n]
        b = bList[i%(n*n)//n]
        if orderByDemand:
            a = pairsByDemand[(i-1)%(n*(n-1)//2)][0]
            b = pairsByDemand[(i-1)%(n*(n-1)//2)][1]
        if a != b and planes[a][b]:
            bestC = -1
            currCost = cost(planes[a][b], distances[a][b])
            for c in range(n):
                if c != a and c != b and distances[a][c] + distances[b][c] < 2*distances[a][b]:
                    newCost = (cost(planes[a][b]+planes[a][c], distances[a][c]) +
                        cost(planes[a][b]+planes[b][c], distances[b][c]) - 
                        cost(planes[a][c], distances[a][c]) - 
                        cost(planes[b][c], distances[b][c]))
                    if newCost < currCost:
                        currCost = newCost 
                        bestC = c 
            if bestC > -1:
                planes[a][bestC] += planes[a][b]
                planes[bestC][a] += planes[a][b]
                planes[b][bestC] += planes[a][b]
                planes[bestC][b] += planes[a][b]
                planes[a][b] = 0
                planes[b][a] = 0
                changes += 1

        if i%smallUpdate == 0:
            if printUpdates:
                print(str(i))
                print(str(changes) + ' changes')
        if i%update == 0:
            totalCost = sum([sum([cost(planes[i][j], distances[i][j]) for i in range(n)]) for j in range(n)])
            if printUpdates:
                print(totalCost)
            costPlot.append(totalCost)
            distPlot.append(np.log(np.sum(planes)))
            if i/update>2:
                if costPlot[-1] == costPlot[-2]:
                    break
                if changes < 1:
                    break
            changes = 0
    finalCost = costPlot[-1]
    print(finalCost)
    if finalCost < bestFinalCost:
        bestFinalCost = finalCost
        bestPlanes = planes 
        bestPorts = ports
        bestCostPlot = costPlot
    
#plt.plot(np.log(bestCostPlot - bestFinalCost))
#plt.show()
if doOpt:
    plot(bestPlanes, optPlanes)
else:
    plot(bestPlanes)

# print(bestPlanes)
# print(optPlanes)
print(bestFinalCost)
print(optScore)


