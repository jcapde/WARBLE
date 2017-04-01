class Tweet_SCAN:
    def __init__(self):
        return
    def run(self, A, B, C, D, Eps1, Eps2, Eps3, MinPts, uparam=0.5):
        def kldiv(doc0, doc1):
            accum=0
            import math
            for i in xrange(len(doc0)):
                if (doc0[i] != 0.0) & (doc1[i] != 0.0):
                    tmp = doc0[i]*math.log(doc0[i]/doc1[i])/math.log(2)
                elif doc0[i] == 0.0:
                    tmp = 0
                else:
                    tmp = float("inf")
                accum += tmp
            return accum

        def js_simil(doc0, doc1):
            import math
            m = 1./2*(doc0+doc1)
            tmp = 1./2*(kldiv(doc0,m)+kldiv(doc1,m))
            if tmp>0:
                aux = math.sqrt(tmp)
            else:
                aux = 0.
            return aux

        def topic_diff(X, o,Eps3):
            ind = []
            for t in X.keys():
                dist = js_simil(o,X[t])
                if dist <= Eps3:
                    ind.append(t)
            return ind

        import numpy as np
        from sklearn.neighbors import NearestNeighbors


        #Initialize variables
        D['class'] = -1
        n = D.user_id.count()
        cluster = 0

        neighbors_model = NearestNeighbors(radius=Eps1, algorithm='auto', leaf_size=30, metric="euclidean")
        neighbors_model.fit(A)
        neighbors_model2 = NearestNeighbors(radius=Eps2, algorithm='auto', leaf_size=30, metric="euclidean")

        for i in xrange(n):
            #print str(i)+" de "+str(n)
            #If the point is in a cluster continue to next point
            if D.ix[i,'class'] != -1:
                continue

            #Retrieve point neighbors
            iNeighbors = neighbors_model.radius_neighbors(A[i].reshape(1,2), Eps1, return_distance=False)[0]
            neighbors_model2.fit(B[iNeighbors].reshape(len(iNeighbors),1))
            tmpn = neighbors_model2.radius_neighbors(B[i].reshape(1,1), Eps2, return_distance=False)[0]
            iNeighbors = iNeighbors[tmpn]
            iNeighbors = topic_diff({j:C[j] for j in iNeighbors}, C[i], Eps3)
            #If the number of neighbors points is less than the minimum points, set noise and continue
            if len(iNeighbors) < MinPts or len(set(D.ix[iNeighbors,"user_id"])) < uparam*len(D.ix[iNeighbors,"user_id"]):
                D.ix[i,'class'] = -1
                continue

            D.ix[i,"class"] = cluster
            candidates = iNeighbors
            while len(candidates)>0:
                c = candidates.pop()
                if D.ix[c,'class'] == -1:
                    D.ix[c,"class"] = cluster
                    nNeighbors = neighbors_model.radius_neighbors(A[c].reshape(1,2),Eps1,return_distance=False)[0]
                    neighbors_model2.fit(B[nNeighbors].reshape(len(nNeighbors),1))
                    tmpn = neighbors_model2.radius_neighbors(B[c].reshape(1,1),Eps2,return_distance=False)[0]
                    nNeighbors = nNeighbors[tmpn]
                    nNeighbors = topic_diff({j:C[j] for j in np.asarray(nNeighbors)},C[c],Eps3)
                    if len(nNeighbors) >= MinPts or len(set(D.ix[nNeighbors,"user_id"])) >= uparam*len(D.ix[nNeighbors,"user_id"]):
                        candidates.extend(nNeighbors)
                        candidates = list(set(candidates))
            cluster += 1
        return D

