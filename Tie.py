list G, P
data = scipy.io.loadmat('example_G.mat')
G.append(data["G"])
data = scipy.io.loadmat('example_G2.mat')
G.append(data["G"])
data = scipy.io.loadmat('example_G3.mat')
G.append(data["G"])
mdic = {"G1": G[0], "G2": G[1], "G3": G[2]}
savemat("exampleG.mat", mdic)

data = scipy.io.loadmat('example_P.mat')
P.append(data["P"])
data = scipy.io.loadmat('example_P2.mat')
P.append(data["P"])
data = scipy.io.loadmat('example_P3.mat')
P.append(data["P"])
mdic = {"P1": P[0], "P2": P[1], "P3": P[2]}