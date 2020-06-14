import math

class D3Coordor(object):
    
    def __init__(self):
        pass

    def dot(self, pos):
        return pos.x * pos.x + pos.y * pos.y + pos.z * pos.z

    # Convert Orthogonal, Sphere and Geographic Coordinate System one to one
    def OrthToSphere(self, pos):
        posDot = self.dot(pos)
        radial = math.sqrt(posDot)
        polar = math.acos(pos.y / radial)
        if pos.z == 0:
            if pos.x >= 0:
                azim = HalfPI
            else:
                azim = -HalfPI
        else:
            azim = math.atan(pos.x / pos.z)
            if pos.z < 0:
                if pos.x >= 0:
                    azim = MPI + azim
                else:
                    azim = MPI - azim
        return SphereCoord(radial, polar, azim)

    def SphereToOrth(self, pos):
        y = pos.radial * math.cos(pos.polar)
        z = pos.radial * math.sin(pos.polar) * math.cos(pos.azim)
        x = pos.radial * math.sin(pos.polar) * math.sin(pos.azim)
        x = float(int(x * 1000000000000000)) / 1000000000000000
        y = float(int(y * 1000000000000000)) / 1000000000000000
        z = float(int(z * 1000000000000000)) / 1000000000000000
        return OrthCoord(x, y, z)

    def SphereToGeo(self, pos):
        lat = 90 - pos.polar * RAD2DEG
        lng = pos.azim * RAD2DEG
        return GeoCoord(lat, lng)

    def GeoToSphere(self, pos):
        polar = (90 - pos.lat) * DEG2RAD
        azim = pos.lng * DEG2RAD
        return SphereCoord(1.0, polar, azim)

    def OrthToGeo(self, pos):
        sprPos = self.OrthToSphere(pos)
        return self.SphereToGeo(sprPos)

    def GeoToOrth(self, pos):
        sprPos = self.GeoToSphere(pos)
        return self.SphereToOrth(sprPos)

class Calculator(object):
    
    def __init__(self):
        self.keyCoder = KeyCoder()
        self.octEarth = OctEarth()

    def SetLayer(self, layer):
        self.octEarth.SetLayer(layer)

    def HexCellKey(self, geoPos):
        index = self.octEarth.HexIndex(geoPos, True)
        key = self.keyCoder.Encode(index)
        return key

 
    def HexCellNeighbor(self, hexKey, k):
        index = self.keyCoder.Decode(hexKey)
        neighbor = self.octEarth.NearestNeighbor(index, k)
        neighborkeys = self.keyCoder.Encodes(neighbor)
        return neighborkeys

    def HexCellBoudary(self, hexKey, k):
        index = self.keyCoder.Decode(hexKey)
        boudary = self.octEarth.NeighborLayer(index, k)
        boudarykeys = self.keyCoder.Encodes(boudary)
        return boudarykeys


    def HexCellVertexesAndCenter(self, hexKey):
        dgrid = self.keyCoder.Decode(hexKey)
        if not dgrid:
            return [], None
        hcg = HexCellGenerator()
        hcg.SetLayer(dgrid.layer)
        vertexes = hcg.HexCellVertexes(dgrid)
        center = hcg.HexCellCenter(dgrid)
        return vertexes, center
    
RADIUS = 6371004
MPI = 3.141592653589793238462643383279502884197169399
HalfPI = MPI / 2
RAD2DEG = 180 / MPI
DEG2RAD = MPI / 180
POLARLAT = 89.999999

VERTEX0DD = "Vertex0DD"
VERTEX1RD = "Vertex1RD"
VERTEX2RU = "Vertex2RU"
VERTEX3UU = "Vertex3UU"
VERTEX4LU = "Vertex4LU"
VERTEX5LD = "Vertex5LD"

class DGrid(object):
    def __init__(self, layer, face, i, j):
        self.layer = layer
        self.face = face
        self.i = i
        self.j = j

    def DGridStr(self):
        return 'OL{0}F{1}i{2}j{3}'.format(self.layer, self.face, self.i, self.j)

class GeoCoord(object):
    
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng

    def GeoCoordStr(self):
        return 'GeoCoord_{0}_lat{1}_lng'.format(self.lat, self.lng)

class OrthCoord(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def OrthCoordStr(self):
        return 'OrthCoord_{0}_x{1}_y{2}_z'.format(self.x, self.y, self.z)

class SphereCoord(object):
    def __init__(self, r, p, a):
        self.radial = r
        self.polar = p
        self.azim = a

    def SphereCoordStr(self):
        return 'SphereCoord_{0}_radial{1}_polar{2}_azim'.format(self.radial, self.polar, self.azim)

class ObliqueCoord(object):
    def __init__(self, i, j, k):
        self.i = i
        self.j = j
        self.k = k

    def ObliqueCoordStr(self):
        return 'ObliqCoord_{0}_i{1}_j{2}_k'.format(self.i, self.j, self.k)

class FaceIndexCoord(object):
    def __init__(self, i, j):
        self.i = i
        self.j = j

    def FaceIndexCoordStr(self):
        return 'FaceIndexCoord_{0}_i{1}_j'.format(self.i, self.j)
class HexCellGenerator(object):
    def __init__(self):
        self.vertex_offset = {}
        self.D3Coord = D3Coordor()
        self.topAxis = OrthCoord(0, 1, 0)
        self.ONETHIRD = 1.0 / 3
        self.vertex_offset[VERTEX0DD] = FaceIndexCoord(self.ONETHIRD, self.ONETHIRD)
        self.vertex_offset[VERTEX1RD] = FaceIndexCoord(-self.ONETHIRD, 2*self.ONETHIRD)
        self.vertex_offset[VERTEX2RU] = FaceIndexCoord(-2*self.ONETHIRD, self.ONETHIRD)
        self.vertex_offset[VERTEX3UU] = FaceIndexCoord(-self.ONETHIRD, -self.ONETHIRD)
        self.vertex_offset[VERTEX4LU] = FaceIndexCoord(self.ONETHIRD, -2*self.ONETHIRD)
        self.vertex_offset[VERTEX5LD] = FaceIndexCoord(2*self.ONETHIRD, -self.ONETHIRD)

    def getOrthPos(self, faceIndex):
        x = faceIndex.i*self.leftAxis.x + faceIndex.j*self.rightAxis.x + self.topAxis.x
        y = faceIndex.i*self.leftAxis.y + faceIndex.j*self.rightAxis.y + self.topAxis.y
        z = faceIndex.i*self.leftAxis.z + faceIndex.j*self.rightAxis.z + self.topAxis.z
        return OrthCoord(x, y, z)
    

    def getFaceIndex(self, hexGrid, vertex_key):
        i = float(hexGrid.i) + self.vertex_offset[vertex_key].i
        j = float(hexGrid.j) + self.vertex_offset[vertex_key].j
        return FaceIndexCoord(i, j)

    def vertexes_polar(self, hexGrid):
        vertexes = [0]*4
        faceID = int(hexGrid.face)
        if hexGrid.i == 0 and hexGrid.j == 0:
            index = self.getFaceIndex(hexGrid, VERTEX0DD)
            orthPos = self.getOrthPos(index)
            geoPos = self.D3Coord.OrthToGeo(orthPos)
            geoPos = self.Topology(geoPos, faceID)
            vertexes[0] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Mirror(geoPos, True, 1)
            vertexes[1] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Mirror(geoPos, True, 1)
            vertexes[2] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Mirror(geoPos, True, 1)
            vertexes[3] = GeoCoord(geoPos.lat, geoPos.lng)
        elif hexGrid.i != 0:
            index = self.getFaceIndex(hexGrid, VERTEX2RU)
            orthPos = self.getOrthPos(index)
            geoPos = self.D3Coord.OrthToGeo(orthPos)
            geoPos = self.Topology(geoPos, faceID)
            vertexes[0] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Mirror(geoPos, False, 1)
            vertexes[1] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Flip(geoPos)
            vertexes[2] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Mirror(geoPos, True, 1)
            vertexes[3] = GeoCoord(geoPos.lat, geoPos.lng)
        else:
            index = self.getFaceIndex(hexGrid, VERTEX4LU)
            orthPos = self.getOrthPos(index)
            geoPos = self.D3Coord.OrthToGeo(orthPos)
            geoPos = self.Topology(geoPos, faceID)
            vertexes[0] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Flip(geoPos)
            vertexes[1] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Mirror(geoPos, True, 1)
            vertexes[2] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Flip(geoPos)
            vertexes[3] = GeoCoord(geoPos.lat, geoPos.lng)
        return vertexes
    

    def vertexes_edge(self, hexGrid):
        vertexes = [0]*6
        faceID = int(hexGrid.face)
        if hexGrid.j == 0:
            index = self.getFaceIndex(hexGrid, VERTEX0DD)
            orthPos = self.getOrthPos(index)
            geoPos = self.D3Coord.OrthToGeo(orthPos)
            geoPos = self.Topology(geoPos, faceID)
            vertexes[0] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Mirror(geoPos, False, 1)
            vertexes[5] = GeoCoord(geoPos.lat, geoPos.lng)

            index = self.getFaceIndex(hexGrid, VERTEX1RD)
            orthPos = self.getOrthPos(index)
            geoPos = self.D3Coord.OrthToGeo(orthPos)
            geoPos = self.Topology(geoPos, faceID)
            vertexes[1] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Mirror(geoPos, False, 1)
            vertexes[4] = GeoCoord(geoPos.lat, geoPos.lng)

            index = self.getFaceIndex(hexGrid, VERTEX2RU)
            orthPos = self.getOrthPos(index)
            geoPos = self.D3Coord.OrthToGeo(orthPos)
            geoPos = self.Topology(geoPos, faceID)
            vertexes[2] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Mirror(geoPos, False, 1)
            vertexes[3] = GeoCoord(geoPos.lat, geoPos.lng)
        elif hexGrid.i == 0:
            index = self.getFaceIndex(hexGrid, VERTEX0DD)
            orthPos = self.getOrthPos(index)
            geoPos = self.D3Coord.OrthToGeo(orthPos)
            geoPos = self.Topology(geoPos, faceID)
            vertexes[5] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Mirror(geoPos, True, 1)
            vertexes[0] = GeoCoord(geoPos.lat, geoPos.lng)

            index = self.getFaceIndex(hexGrid, VERTEX5LD)
            orthPos = self.getOrthPos(index)
            geoPos = self.D3Coord.OrthToGeo(orthPos)
            geoPos = self.Topology(geoPos, faceID)
            vertexes[4] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Mirror(geoPos, True, 1)
            vertexes[1] = GeoCoord(geoPos.lat, geoPos.lng)

            index = self.getFaceIndex(hexGrid, VERTEX4LU)
            orthPos = self.getOrthPos(index)
            geoPos = self.D3Coord.OrthToGeo(orthPos)
            geoPos = self.Topology(geoPos, faceID)
            vertexes[3] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Mirror(geoPos, True, 1)
            vertexes[2] = GeoCoord(geoPos.lat, geoPos.lng)
        else:
            index = self.getFaceIndex(hexGrid, VERTEX2RU)
            orthPos = self.getOrthPos(index)
            geoPos = self.D3Coord.OrthToGeo(orthPos)
            geoPos = self.Topology(geoPos, faceID)
            vertexes[2] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Flip(geoPos)
            vertexes[1] = GeoCoord(geoPos.lat, geoPos.lng)

            index = self.getFaceIndex(hexGrid, VERTEX3UU)
            orthPos = self.getOrthPos(index)
            geoPos = self.D3Coord.OrthToGeo(orthPos)
            geoPos = self.Topology(geoPos, faceID)
            vertexes[3] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Flip(geoPos)
            vertexes[0] = GeoCoord(geoPos.lat, geoPos.lng)

            index = self.getFaceIndex(hexGrid, VERTEX4LU)
            orthPos = self.getOrthPos(index)
            geoPos = self.D3Coord.OrthToGeo(orthPos)
            geoPos = self.Topology(geoPos, faceID)
            vertexes[4] = GeoCoord(geoPos.lat, geoPos.lng)
            geoPos = self.Flip(geoPos)
            vertexes[5] = GeoCoord(geoPos.lat, geoPos.lng)
        return vertexes
    

    def vertexes_inner(self, hexGrid):
        vertexes =  []
        faceID = int(hexGrid.face)
        VERLIST = [VERTEX0DD, VERTEX1RD, VERTEX2RU, VERTEX3UU, VERTEX4LU, VERTEX5LD]
        for k in VERLIST:
            index = self.getFaceIndex(hexGrid, k)
            orthPos = self.getOrthPos(index)
            geoPos = self.D3Coord.OrthToGeo(orthPos)
            geoPos = self.Topology(geoPos, faceID)
            vertexes.append(geoPos)
        return vertexes
    

    def SetLayer(self, layer):
        self.layer = layer
        self.N = int(3 * pow(2.0, (float)(layer-1)))
        self.leftAxis = OrthCoord(0, -1.0/(float)(self.N), 1.0/(float)(self.N))
        self.rightAxis = OrthCoord(1.0/(float)(self.N), -1.0/(float)(self.N), 0)
    
    def FaceID(self, geoPos):
        lat = geoPos.lat
        lon = geoPos.lng
        if lon >= 0:
            if lon <= 90:
                index = 0
            else:
                index = 1
        else:
            if lon <= -90:
                index = 2
            else:
                index = 3
        if lat < 0:
            index = index + 4
        return index
    
    def ExchangeIJ(self, geoPos):
        if geoPos.lng >= 0:
            if geoPos.lng < 90:
                geoPos.lng = 90 - geoPos.lng
            else:
                geoPos.lng = 270 - geoPos.lng
        else:
            if geoPos.lng > -90:
                geoPos.lng = -90 - geoPos.lng
            else:
                geoPos.lng = -270 - geoPos.lng
        return geoPos
    

    def Flip(self, geoPos):
        geoPos.lat = -geoPos.lat
        return geoPos
    

    def Rotate(self, geoPos, westToEast, nTimes):
        if nTimes == 1:
            if westToEast:
                geoPos.lng = geoPos.lng + 90
                if geoPos.lng > 180:
                    geoPos.lng = geoPos.lng - 360
            else:
                geoPos.lng = geoPos.lng - 90
                if geoPos.lng < -180:
                    geoPos.lng = geoPos.lng + 360
        else:
            for i in range(nTimes):
                geoPos = self.Rotate(geoPos, westToEast, 1)
        return geoPos

    def Mirror(self, geoPos, westToEast, nTimes):
        geoPos = self.Rotate(geoPos, westToEast, nTimes)
        if nTimes%2 == 1:
            geoPos = self.ExchangeIJ(geoPos)
        return geoPos

    def Symmetry(self, geoPos):
        geoPos = self.Flip(geoPos)
        return self.Rotate(geoPos, True, 2)

    def Topology(self, geoPos, faceID):
        if faceID >= 4:
            faceID = faceID - 4
            geoPos = self.Flip(geoPos)
        return self.Rotate(geoPos, True, faceID)
    
    def HexCellVertexes(self, hexGrid):
        if (hexGrid.i == 0 and hexGrid.j == 0) or (hexGrid.i == self.N and hexGrid.j == 0) or (hexGrid.i == 0 and hexGrid.j == self.N):
            return self.vertexes_polar(hexGrid)
        elif (hexGrid.i == 0 or hexGrid.j == 0) or (hexGrid.i+hexGrid.j == self.N):
            return self.vertexes_edge(hexGrid)
        else:
            return self.vertexes_inner(hexGrid)

    def HexCellCenter(self, hexGrid):
        index = FaceIndexCoord(float(hexGrid.i), float(hexGrid.j))
        orthPos = self.getOrthPos(index)
        geoPos = self.D3Coord.OrthToGeo(orthPos)
        geoPos = self.Topology(geoPos, int(hexGrid.face))
        return geoPos
    
KEYLEN=4
class HexPicker(object):
    def __init__(self):
        self.keylen = KEYLEN

    def ExchangeIJ(self, dg):
        tmp = dg.i
        dg.i = dg.j
        dg.j = tmp
        return dg

    def FlipFaceId(self, faceID):
        if faceID > 3:
            faceID = faceID - 4
        elif faceID < 4:
            faceID = faceID + 4
        return faceID

    def FlipDGrid(self, dg):
        return self.FlipFaceId(dg.face)

    def RotateFaceInd(self, faceInd, westToEast, nTimes):
        if nTimes == 1:
            if westToEast:
                faceInd = faceInd + 1
                if faceInd == 4:
                    faceInd = 0
                
                if faceInd == 8:
                    faceInd = 4
            else:
                faceInd = faceInd - 1
                if faceInd == -1:
                    faceInd = 3
                elif faceInd == 3:
                    faceInd = 7
        else:
            for i in range(nTimes):
                faceInd = self.RotateFaceInd(faceInd, westToEast, 1)
        return faceInd

    def RotateDGrid(self, dg, westToEast, nTimes):
        faceInd = self.RotateFaceInd(dg.face, westToEast, nTimes)
        dg.face = faceInd
        return dg
    
    def MirrorFaceInd(self, faceInd, westToEast, nTimes):
        return self.RotateFaceInd(faceInd, westToEast, nTimes)

    def MirrorDGrid(self, dg, westToEast, nTimes):
        dg = self.RotateDGrid(dg, westToEast, nTimes)
        if nTimes%2 == 1:
            dg = self.ExchangeIJ(dg)
        return dg
    
    def SymmetryFaceInd(self, faceInd):
        faceInd = self.FlipFaceId(faceInd)
        return self.RotateFaceInd(faceInd, True, 2)
    
    def SymmetryDGrid(self, dg):
        return self.SymmetryFaceInd(dg.face)

    def UniqueIndex(self, indexes):
       
        newIndexes = []
       
        s = {}
        for v in indexes:
            key = v.DGridStr()
            if key not in s:
                newIndexes.append(v)
                s[key] = True
        return newIndexes
class KeyCoder(object):
    def __init__(self):
        pass

   
    def Encode(self, hexIndex):
        return 'OL{0}F{1}i{2}j{3}'.format(hexIndex.layer, hexIndex.face, hexIndex.i, hexIndex.j)

  
    def Encodes(self, hexIndexes):
        s = []
        for hexIndex in hexIndexes:
            s.append(self.Encode(hexIndex))
        return s

  
    def Decode(self, hexKey):
        lpos = hexKey.find("L")
        if lpos == -1:
            return False
        fpos = hexKey.find("F")
        if fpos == -1:
            return False
        ipos = hexKey.find("i")
        if ipos == -1:
            return False
        jpos = hexKey.find("j")
        if jpos == -1:
            return False
        size = len(hexKey)
        lind = int(hexKey[lpos+1 : fpos])
        find = int(hexKey[fpos+1 : ipos])
        iind = int(hexKey[ipos+1 : jpos])
        jind = int(hexKey[jpos+1 : size])
        return DGrid(lind, find, iind, jind)

    
    def Decodes(self, hexKeys):
        dgs = []
        for hexKey in hexKeys:
            dgs.append(self.Decode(hexKey))
        return dgs
class OctEarth(object):
    def __init__(self):
        self.n = None
        self.cellLength = None
        self.layer = None
        self.hexPicker = HexPicker()
        self.d3Coord = D3Coordor()

    def isNearest(self, ii, jj):
        a = ii + jj/2
        b = jj + ii/2
        if a <= 0.5 and b <= 0.5:
            return True
        return False

    def SetLayer(self, layer):
        self.layer = layer
        self.n = int(3 * pow(2.0, (float)(layer-1)))

    def FaceID(self, geoPos):
        lat = geoPos.lat
        lon = geoPos.lng
        if lon >= 0:
            if lon <= 90:
                index = 0
            else:
                index = 1
        else:
            if lon <= -90:
                index = 2
            else:
                index = 3
        if lat < 0:
            index = index + 4
        return index

    def FacePos(self, geoPos):
        if geoPos.lng >= 0:
            if geoPos.lng > 90:
                geoPos.lng = geoPos.lng - 90
        else:
            if geoPos.lng <= -90:
                geoPos.lng = geoPos.lng + 180
            else:
                geoPos.lng = geoPos.lng + 90
        if geoPos.lat < 0:
            geoPos.lat = -geoPos.lat
        return geoPos

    def ObliqueCoord(self, geoPos):
        orthPos = self.d3Coord.GeoToOrth(geoPos)
        s = orthPos.x + orthPos.y + orthPos.z
        k = (s - 1) / s
        j = orthPos.x * float(1 - k) * float(self.n)
        i = orthPos.z * float(1 - k) * float(self.n)
        return ObliqueCoord(i, j, k)

    def FaceCoord(self, geoPos):
        coord = self.ObliqueCoord(geoPos)
        return FaceIndexCoord(coord.i, coord.j)

    def NearestIndex(self, index):
        i = math.floor(index.i)
        j = math.floor(index.j)
        ii = index.i - i
        jj = index.j - j
        if self.isNearest(ii, jj):
            fi = i
            fj = j
        elif self.isNearest(1-ii, 1-jj):
            fi = i + 1
            fj = j + 1
        elif ii > jj:
            fi = i + 1
            fj = j
        else:
            fi = i
            fj = j + 1
        return FaceIndexCoord(fi, fj)

    def HexIndex(self, geoPos, reduceSameIndex):
        frameIndex = self.FaceID(geoPos)
        geoPos = self.FacePos(geoPos)
        index = self.FaceCoord(geoPos)
        nstIndex = self.NearestIndex(index)
        hexIndex = DGrid(self.layer, frameIndex, int(nstIndex.i), int(nstIndex.j))
        if reduceSameIndex:
            hexIndex = self.AdjustEdgeHexIndex(hexIndex)
        return hexIndex

    def AdjustEdgeHexIndex(self, hexIndex):
        #South to North
        if hexIndex.i+hexIndex.j == int(self.n):
            if hexIndex.face > 3:
                hexIndex.face = hexIndex.face - 4
        #West to East
        if hexIndex.i == 0:
            hexIndex = self.hexPicker.MirrorDGrid(hexIndex, True, 1)
        #Polar vertex
        if hexIndex.i == 0 and hexIndex.j == 0:
            if hexIndex.face < 4:
                hexIndex.face = 0
            else:
                hexIndex.face = 4
        return hexIndex

    def EffectiveEarthNeighborK(self, k):
        if k < 0:
            k = -k
        k = k % (4 * self.n)
        if k > 2*self.n:
            k = 4*self.n - k
        return k

    def UnfoldNeighbor(self, hexIndex, k):
        neighbor = []
        if k <= 0:
            return neighbor
        kk = int(k)
        ii = int(k)
        for jj in range(0, -kk-1, -1):
            neighbor.append(DGrid(hexIndex.layer, hexIndex.face, hexIndex.i+ii, hexIndex.j+jj))
        jj = -kk
        for ii in range(kk-1, -1, -1):
            neighbor.append(DGrid(hexIndex.layer, hexIndex.face, hexIndex.i+ii, hexIndex.j+jj))
        for ii in range(-1, 1-kk-1, -1):
            jj = -kk - ii
            neighbor.append(DGrid(hexIndex.layer, hexIndex.face, hexIndex.i+ii, hexIndex.j+jj))
        ii = -kk
        for jj in range(0, kk+1):
            neighbor.append(DGrid(hexIndex.layer, hexIndex.face, hexIndex.i+ii, hexIndex.j+jj))
        jj = kk
        for ii in range(1-kk, 1):
            neighbor.append(DGrid(hexIndex.layer, hexIndex.face, hexIndex.i+ii, hexIndex.j+jj))
        for ii in range(1, kk):
            jj = kk - ii
            neighbor.append(DGrid(hexIndex.layer, hexIndex.face, hexIndex.i+ii, hexIndex.j+jj))
        return neighbor

    def TopoFaceID(self, baseFace, topoFace, inverse):
        if baseFace == 0:
            topoInd = topoFace
        elif baseFace == 4:
            topoInd = self.hexPicker.FlipFaceId(topoFace)
        elif baseFace == 1:
            topoInd = self.hexPicker.RotateFaceInd(topoFace, inverse, 1)
        elif baseFace == 5:
            topoInd = self.hexPicker.FlipFaceId(topoFace)
            topoInd = self.hexPicker.RotateFaceInd(topoInd, inverse, 1)
        elif baseFace == 2:
            topoInd = self.hexPicker.RotateFaceInd(topoFace, True, 2)
        elif baseFace == 6:
            topoInd = self.hexPicker.FlipFaceId(topoFace)
            topoInd = self.hexPicker.RotateFaceInd(topoInd, True, 2)
        elif baseFace == 3:
            topoInd = self.hexPicker.RotateFaceInd(topoFace, not inverse, 1)
        elif baseFace == 7:
            topoInd = self.hexPicker.FlipFaceId(topoFace)
            topoInd = self.hexPicker.RotateFaceInd(topoInd, not inverse, 1)
        else:
            topoInd = -1
        return topoInd

    def FoldTopoFaceID(self, unfoldHexInde):
        i = unfoldHexInde.i
        j = unfoldHexInde.j
        n = int(self.n)
        if i >= 0 and i <= n and j >= 0 and j <= n:
            if i+j <= n:
                return 0
            else:
                return 4
        if i > 0 and i <= n and j < 0 and j >= -n:
            if i+j >= 0:
                return 3
            else:
                return 21
        if j > 0 and j <= n and i < 0 and i > -n:
            if i+j >= 0:
                return 1
            else:
                return 22
        if i <= 0 and i >= -n and j <= 0 and j >= -n and i+j >= -n:
            return 20
        if i > n and i <= 2*n and j <= 0 and j >= -n:
            if i+j < n:
                return 71
            else:
                return 70
        if i > n and i < 2*n and j > 0 and j < n and i+j <= 2*n:
            return 72
        if j > n and j <= 2*n and i <= 0 and i >= -n:
            if (i+j) < n:
                return 51
            else:
                return 50
        if j > n and j < 2*n and i > 0 and i < n and i+j <= 2*n:
            return 52
        return -1

    def FoldHexIndex(self, index):
        faceID = self.FoldTopoFaceID(index)
        fldIndex = DGrid(index.layer, index.face, index.i, index.j)
        n = int(self.n)
        if faceID == -1:
            fldIndex.face = faceID
            return fldIndex
        elif faceID == 0:
            pass
        elif faceID == 4:
            fldIndex.i = n - index.j
            fldIndex.j = n - index.i
        elif faceID == 1:
            fldIndex.i = index.i + index.j
            fldIndex.j = -index.i
        elif faceID == 50:
            fldIndex.i = 2*n - index.j
            fldIndex.j = -index.i
            faceID = 5
        elif faceID == 51:
            fldIndex.i = index.i + n
            fldIndex.j = n - index.i - index.j
            faceID = 5
        elif faceID == 52:
            fldIndex.i = 2*n - index.i - index.j
            fldIndex.j = index.j - n
            faceID = 5
        elif faceID == 3:
            fldIndex.i = -index.j
            fldIndex.j = index.i + index.j
        elif faceID == 70:
            fldIndex.i = -index.j
            fldIndex.j = 2*n - index.i
            faceID = 7
        elif faceID == 71:
            fldIndex.i = n - index.i - index.j
            fldIndex.j = index.j + n
            faceID = 7
        elif faceID == 72:
            fldIndex.i = index.i - n
            fldIndex.j = 2 * n - index.i - index.j
            faceID = 7
        elif faceID == 20:
            fldIndex.i = -index.i
            fldIndex.j = -index.j
            faceID = 2
        elif faceID == 21:
            fldIndex.i = -index.i - index.j
            fldIndex.j = index.i
            faceID = 2
        elif faceID == 22:
            fldIndex.i = index.j
            fldIndex.j = -index.i - index.j
            faceID = 2
        fldFaceInd = self.TopoFaceID(index.face, faceID, True)
        fldIndex.face = fldFaceInd
        return fldIndex

    def NeighborLayer(self, hexIndex, k):
        k = self.EffectiveEarthNeighborK(k)
        if k > self.n:
            hexIndex = self.hexPicker.SymmetryDGrid(hexIndex)
            symK = 2*self.n - k
            return self.NeighborLayer(hexIndex, symK)
        neighbor = []
        if k == 0:
            neighbor.append(DGrid(hexIndex.layer, hexIndex.face, hexIndex.i, hexIndex.j))
            return neighbor
        ufldNeighbor = self.UnfoldNeighbor(hexIndex, k)
        if hexIndex.i > int(k) and hexIndex.j > int(k) and (int(self.n-hexIndex.i-hexIndex.j)) > int(k):
            return ufldNeighbor
        for v in ufldNeighbor:
            fldIndex = self.FoldHexIndex(v)
            if fldIndex.face != -1:
                fldIndex = self.AdjustEdgeHexIndex(fldIndex)
                neighbor.append(fldIndex)
        return self.hexPicker.UniqueIndex(neighbor)

    def NearestNeighbor(self, hexIndex, k):
        newK = self.EffectiveEarthNeighborK(k)
        neighbor = []
        for i in range(newK+1):
            iNeighbor = self.NeighborLayer(hexIndex, i)
            for v in iNeighbor:
                neighbor.append(v)
        return self.hexPicker.UniqueIndex(neighbor)