import numpy as np
# from sklearn.preprocessing import MinMaxScaler
"""
1. Normal pattern: y(t) = m + rs (2)
Where m = 30, s= 2 and r is a random number between ±3
2. Cyclic pattern: y(t) = m + rs + aSIN(2πt/T) (3)
Where a and T take values between 10 and 15 for each pattern
3. Increasing shift: y(t) = m + rs +gt (4)
Where g takes a value between 0.2 and 0.5 for each pattern
4. Decreasing shift: y(t) = m + rs – gt (5)
5. Upward shift: y(t) = m + rs + kx (6)
Where, for each pattern, x takes a value between 7.5 and 20. k = 0 before time t3 and 1 after this
time. t3 takes a value between n/3 and 2n/3 for each pattern
6. Downward shift: y(t) = m + rs – kx (7) 

"""
np.random.seed(42)
# scaler = MinMaxScaler(feature_range=(-20, 20))
# data = []
# for l in range(100,1100,100):
l=100
n = 200
# l = 100
def generateSyntheticData(_l,n):
    ### normal ###
    m = 30
    s = 2
    l = np.random.randint(1,_l)
    r = np.random.uniform(low=-1, high=1, size=(l,n))
    normal = m + r*s
    normal = normal - np.mean(normal,axis=1).reshape(-1,1)
    ### normal ###

    ### cyclic ###
    l = np.random.randint(1,_l)
    r = np.random.uniform(low=-3, high=3, size=(l,n))
    T = np.random.uniform(low=10.0,high=15.0,size=(l,))
    T = np.tile(T,(n,1)).transpose()
    a = np.random.uniform(low=10.0,high=15.0,size=(l,))
    a = np.tile(a,(n,1)).transpose()

    t = np.tile(np.arange(n),(l,1))
    cyclic = m + r*s + a * np.sin(t * 2 * np.pi/T)
    cyclic = cyclic - np.mean(cyclic,axis=1).reshape(-1,1)
    ### cyclic ###

    ### increasing shift ###
    l = np.random.randint(1,_l)
    t = np.tile(np.arange(n),(l,1))
    r = np.random.uniform(low=-3, high=3, size=(l,n))
    g = np.random.uniform(low=0.4,high=0.5,size=(l,))
    g = np.tile(g,(n,1)).transpose()
    increasing_shift = m + r*s + g*t
    # increasing_shift = scaler.fit_transform(increasing_shift.transpose()).transpose()#
    increasing_shift = increasing_shift - np.mean(increasing_shift,axis=1).reshape(-1,1)
    ### increasing shift ###

    ### decreasing shift ###
    l = np.random.randint(1,_l)
    t = np.tile(np.arange(n),(l,1))
    r = np.random.uniform(low=-3, high=3, size=(l,n))
    g = np.random.uniform(low=0.4,high=0.5,size=(l,))
    g = np.tile(g,(n,1)).transpose()
    decreasing_shift = m + r*s - g*t
    # decreasing_shift = scaler.fit_transform(decreasing_shift.transpose()).transpose()
    decreasing_shift = decreasing_shift - np.mean(decreasing_shift,axis=1).reshape(-1,1)
    ### decreasing shift ###

    ### upward_shift ###
    l = np.random.randint(1,_l)
    r = np.random.uniform(low=-3, high=3, size=(l,n))
    x = np.random.uniform(low=7.5,high=20,size=(l,))
    x = np.tile(x,(n,1)).transpose()
    _t = np.random.uniform(low=n/3,high=2*n/3,size=(l,))
    _t = np.tile(_t,(n,1)).transpose()
    t = np.tile(np.arange(n),(l,1))
    k = np.zeros((l,n))
    k[t>=_t]=1
    upward_shift = m + r*s + k*x
    upward_shift = upward_shift - np.mean(upward_shift,axis=1).reshape(-1,1)
    ### upward_shift ###

    ### downward_shift ###
    l = np.random.randint(1,_l)
    r = np.random.uniform(low=-3, high=3, size=(l,n))
    x = np.random.uniform(low=7.5,high=20,size=(l,))
    x = np.tile(x,(n,1)).transpose()
    _t = np.random.uniform(low=n/3,high=2*n/3,size=(l,))
    _t = np.tile(_t,(n,1)).transpose()
    t = np.tile(np.arange(n),(l,1))
    k = np.zeros((l,n))
    k[t>=_t]=1
    downward_shift = m + r*s - k*x
    downward_shift = downward_shift - np.mean(downward_shift,axis=1).reshape(-1,1)
    ### downward_shift ###

    data = [decreasing_shift,increasing_shift,downward_shift,cyclic,upward_shift] #np.concatenate((decreasing_shift,increasing_shift,downward_shift,cyclic,upward_shift),axis=0)
    true_labels = np.repeat([0,1,2,3,4],l, axis=0)
    return data