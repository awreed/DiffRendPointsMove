import numpy as np

class ObjLoader(object):
    def __init__(self, fileName):
        self.vertices = []
        self.faces = []
        self.centroids = []
        ##
        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                    self.vertices.append(vertex)

                elif line[0] == "f":
                    string = line.replace("//", "/")
                    ##
                    i = string.find(" ") + 1
                    face = []
                    for item in range(string.count(" ")):
                        if string.find(" ", i) == -1:
                            face.append(string[i:-1])
                            break
                        face.append(string[i:string.find(" ", i)])
                        i = string.find(" ", i) + 1
                    ##
                    self.faces.append(tuple(face))
            self.vertices = np.asarray(self.vertices)
            self.faces = np.asarray(self.faces)
            f.close()
        except IOError:
            print(".obj file not found.")

    def getCentroids(self):
        vs = self.vertices
        for f in self.faces:
            v1 = f[0].split('/')[0]
            v2 = f[1].split('/')[0]
            v3 = f[2].split('/')[0]
            med = (vs[int(v1)-1, :] + vs[int(v2)-1, :] + vs[int(v3)-1, :])/3
            med = np.array(med)
            #med = np.array([med[0], med[2]])
            self.centroids.append(med)

        self.centroids = np.asarray(self.centroids)
        #print(self.centroids)
        # Do this if you only want unique 2D points
        #new_array = [tuple(row) for row in self.centroids]
        #self.centroids = np.unique(new_array, axis=0)
        #print(self.centroids)