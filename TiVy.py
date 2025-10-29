import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse import csr_matrix,triu
from scipy.cluster.hierarchy import fcluster,linkage, leaves_list, optimal_leaf_ordering
import math
from scipy.spatial.distance import euclidean
from dtaidistance import dtw
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy import stats
from joblib import Parallel, delayed
from copy import deepcopy
import time
import json
from random import randint
from scipy.spatial.distance import squareform, pdist


from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tslearn.metrics import SoftDTWLossPyTorch
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances

NAN_NUMBER = 0.000000001
np.random.seed(42)

from scipy.spatial.distance import pdist
from dtaidistance import dtw
from sklearn.preprocessing import minmax_scale

def get_dist(clusters,data):
    def get_medoid(data,c):
        subset = data[c[0],c[1]:c[2]]
        random_indice = np.arange(subset.shape[0])#np.random.choice(np.arange(subset.shape[0]), min(100,subset.shape[0]), replace=False)
        _centroid = subset[random_indice][get_average_sequence(subset[random_indice]),:]
        return _centroid

    medoids = []
    avg_shape_dists = []
    temporal_spreads = []

    cluster_dists = []
    temporal_overlaps = []
    
    for cluster in clusters:
        values = data[cluster[0],cluster[1]:cluster[2]]
        medoids.append(get_medoid(data,cluster))
        avg_shape_dists.append(np.mean(dtw.distance_matrix_fast(values)))
        temporal_spreads.append(cluster[2] - cluster[1])

    # s = len(medoids) * (len(values)+1) / 2
    cluster_dists = squareform(dtw.distance_matrix_fast(medoids))    
    temporal_overlaps = pdist([[x[1],x[2]] for x in clusters],lambda x,y: max(0, min(x[1], y[1]) - max(x[0], y[0]))) 

    return avg_shape_dists, temporal_spreads, cluster_dists, temporal_overlaps 

def cluster_evaluation(clusters_combinations,data):
    start_idx = {'avg_shape_dists':[],'temporal_spreads':[],'cluster_dists':[],'temporal_overlaps':[]}
    avg_shape_dists = [] 
    temporal_spreads = [] 
    cluster_dists = []
    temporal_overlaps = []

    for clusters in clusters_combinations:
        if len(clusters['result']) > 0:
            result = get_dist(clusters['result'], data)
        else:
            result = [[],[],np.array([]),np.array([])]
        avg_shape_dists += result[0]
        temporal_spreads += result[1]
        cluster_dists += result[2].tolist()
        temporal_overlaps += result[3].tolist()

        i = {k:v[-1][-1] if len(v)>0 else 0 for k,v in start_idx.items()}

        start_idx['avg_shape_dists'].append([i['avg_shape_dists'],len(result[0])+i['avg_shape_dists']])
        start_idx['temporal_spreads'].append([i['temporal_spreads'],len(result[1])+i['temporal_spreads']])
        start_idx['cluster_dists'].append([i['cluster_dists'],len(result[2])+i['cluster_dists']])
        start_idx['temporal_overlaps'].append([i['temporal_overlaps'],len(result[3])+i['temporal_overlaps']])
    
        
    avg_shape_dists = minmax_scale(avg_shape_dists)
    temporal_spreads = minmax_scale(temporal_spreads)
    cluster_dists = minmax_scale(cluster_dists)
    temporal_overlaps = minmax_scale(temporal_overlaps)

    scores = []
    for i,clusters in enumerate(clusters_combinations):
        n_points = 0
        n_clusters = len(clusters['result'])
        for c in clusters['result']:
            n_points += len(c[0]) * (c[2] - c[1])
        
        total_intra = np.sum(avg_shape_dists[start_idx['avg_shape_dists'][i][0]:start_idx['avg_shape_dists'][i][1]] + temporal_spreads[start_idx['temporal_spreads'][i][0]:start_idx['temporal_spreads'][i][1]])
        total_inter = np.sum(cluster_dists[start_idx['cluster_dists'][i][0]:start_idx['cluster_dists'][i][1]] + temporal_overlaps[start_idx['temporal_overlaps'][i][0]:start_idx['temporal_overlaps'][i][1]])
        score = total_inter / (total_intra + 1e-8)
        scores.append(score)
    return np.argmax(scores), np.argsort(-np.array(scores))
    
def get_flat_list(ts):
    n = ts.shape[0]
    p = ts.shape[1]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0])
    dists_cond = squareform(dtw.distance_matrix_fast(ts,window=min(10,int(p*0.1))))
    dists_cond = np.nan_to_num(dists_cond, nan=0)
    Z = linkage(dists_cond, method='ward', metric='euclidean')
    return leaves_list(optimal_leaf_ordering(Z, ts))

def get_average_sequence(ts):
    n = ts.shape[0]
    p = ts.shape[1]
    if p > 10000:
        return np.random.randint(low=0,high=n)
    else:
        A = dtw.distance_matrix_fast(ts)#dtw.distance_matrix_fast(ts,window=min(10,int(p*0.1)),psi = 2)
        A[A==np.inf] = 0
        # A = squareform(A)
        out = A.T + A
        idx = np.arange(A.shape[0])
        out[idx,idx] = A[idx,idx]
        idxAvg = np.argmin(np.sum(out,axis=0))
        return int(idxAvg)

def gap_rule(_X,B=5,max_iter=None,strength=1,step=0.01,distance='dtw',dtw_model=None,device=None):
    def generate_linkage(ts,distance='euclidean'):
        n=ts.shape[0]
        p = ts.shape[1]
        if distance == 'euclidean':
            dists = squareform(pdist(ts))
            
        elif distance == 'dtw':
            dists = dtw.distance_matrix_fast(ts,use_pruning=False,psi = None)
        dists_cond = np.zeros(int((n * (n - 1)) / 2))
        idx = 0
        for r in range(len(ts) - 1):
            dists_cond[idx:idx + len(ts) - r - 1] = dists[r, r + 1:]
            idx += len(ts) - r - 1
        Z = linkage(dists_cond, method='ward', metric='euclidean')
        return Z,dists

    def calculate_w(Z,dist,k):
        N=dist.shape[0]
        labels=fcluster(Z, k, criterion='maxclust')
        unique_labels=list(set(labels))
        w=0
        for l in unique_labels:
            idx=np.where(labels==l)[0]
            pairwise_dist=np.triu(dist[idx,:][:,idx], k=0)
            pairwise_dist[pairwise_dist == np.inf] = 0
            w+=np.sum(np.square(pairwise_dist))/N
        if w != 0:
            w=np.log(w)
        return w,labels

    N = _X.shape[0]
    p = _X.shape[1]
    if N == 1:
        return np.array([1])

    # if normalize == 'mean':
    #     # X = stats.zscore(_X)
    #     X = (_X.transpose() - np.mean(_X,axis=1)).transpose()
    #     X[np.isnan(X)]=0
    # elif normalize == 'zscore':
    #     X = stats.zscore(_X)
    # else:
    X = _X
    #generate reference dataset
    sample_space = []
    for i in range(_X.shape[1]):
        tmp = X[:,i]
        sample_space.append(np.linspace(np.min(tmp),np.max(tmp),N))

    sample_space = np.array(sample_space).transpose()

    Z_sim = []
    distances = []
    #simulation
    for i in range(B):
        sample = np.random.randint(0,N-1,(N,p))
        coord = np.array([range(p) for i in range(N)])
        sample = sample_space[sample,coord]
        # sample = np.dot(sample,vh)
        Z,dists = generate_linkage(sample,distance=distance)
        Z_sim.append(Z)
        distances.append(dists)
    #real
    Z_actual,dists_real = generate_linkage(X,distance=distance)
    k=1
    prev_gap=None
    prev_labels=None
    _iter=N
    if max_iter != None:
        _iter=min(max_iter,_iter)
    gaps=[]
    while k<=_iter:
        #w_exp
        w_exp=[]
        for Z,dist in zip(Z_sim,distances):
            w_exp.append(calculate_w(Z,dist,k)[0])
        w_exp=np.array(w_exp)
        s=np.std(w_exp)*np.sqrt(1+1/B)
        w_exp=w_exp.sum()/B
        #w
        w,labels=calculate_w(Z_actual,dists_real,k)
        gap=w_exp-w
        gaps.append(gap)
        if prev_gap==None:
            prev_gap=gap
            prev_labels=labels
            k+=1
            continue
        else:
            if prev_gap<0:
                diff=prev_gap*(1/strength)-gap+s
            else:
                diff=prev_gap*strength-gap+s
            if diff>=0:
                return prev_labels
        prev_gap=gap
        prev_labels=labels
        strength +=step
        k+=1
    return labels

def create_clusters(args):
    valid_pos = np.where((args['ts']!=NAN_NUMBER).all(axis=1))[0]
    invalid_pos = np.where((args['ts']==NAN_NUMBER).any(axis=1))[0]
    X = args['ts'][valid_pos,:]
    B = args['B']
    max_iter = args['max_iter']
    strength = args['strength']
    step = args['step']
    segment = args['segment']
    cutoff = args['cutoff']
    nPerm = args['nPerm']
    bandWidth = args['bandWidth']
    use_lsh = args['use_LSH']
    LSH_sample = args['LSH_sample']

    if len(X)==0:
        labels_final = np.zeros(args['ts'].shape[0])
        labels_final = labels_final.astype(int)
        return labels_final

    # LSH
    if use_lsh:
        DIM = X.shape[1]
        w = max(1,np.floor(bandWidth * np.sqrt(DIM)))
        b = np.random.uniform(0,w,size=nPerm)
        x = np.random.normal(0,1,size = (DIM,nPerm))
        lsh = np.floor((np.dot(X,x) + b) / w)
        buckets, lsh_index = np.unique(lsh,axis=0,return_inverse=True)
        buckets = np.unique(lsh_index)
    
        labels = np.zeros(args['ts'].shape[0])
        l = 1
        for bucket in buckets:
            _mat = X[lsh_index == bucket,:]
            if _mat.shape[0] == 0:
                continue
            labels[lsh_index == bucket] = l
            l += 1

        core_sets = []
        pre_groups = {}
        labels_unique = np.unique(labels)
        for l in labels_unique:
            pre_groups[l] = labels == l
            # core_sets.append(np.random.choice(np.where(pre_groups[l])[0]))
            core_sets += np.random.choice(np.where(pre_groups[l])[0],LSH_sample).tolist()

        core_labels = gap_rule(X[core_sets,:],B,max_iter,strength,step) 
        def rowwise_mode_bincount(arr, n):
            arr_2d = arr.reshape(-1, n)
            return np.array([np.bincount(row).argmax() for row in arr_2d])

        core_labels = rowwise_mode_bincount(core_labels,LSH_sample)
        core_labels = core_labels
        for u,c in zip(labels_unique,core_labels):
            labels[pre_groups[u]] = c
        #
        labels_final = np.zeros(args['ts'].shape[0])
        labels_final[valid_pos] = labels
        labels_final[invalid_pos] = 0
        labels_final = labels_final.astype(int)
        return labels_final
    else:
        labels = gap_rule(X,B,max_iter,strength,step)
        return labels

def optimize_sequence_by_support(support = 10,counter=None,sequenceCluster=None):
    
    seqByNumber={}
    keys = list(counter.keys())
    for k in keys:
        tlp = eval(k)
        if len(tlp) not in seqByNumber:
            seqByNumber[len(tlp)]=[]#IntervalTree()
        if counter[k] >= support:
            seqByNumber[len(tlp)].append(k)
        else:
            del sequenceCluster[k]
            del counter[k]
    order= sorted(list(seqByNumber.keys()),reverse=True)
    result = []
    for i in order:
        candidates = []
        for val in seqByNumber[i]:#.items():
            if counter[val] >=support:
            # if counter[val.data] >=support:
                candidates.append(val)
            else:
                del sequenceCluster[val]
                del counter[val]

        candidates = list(set(candidates)) 
        while len(candidates) != 0 :
            idx = randint(0,len(candidates)-1)
            c = candidates[idx]
            setRemove = []
            setOverlap = []
            seq = sequenceCluster[c]

            for j in candidates:
                if c == j:
                    continue
                candSeq = sequenceCluster[j]
                if candSeq[1] < seq[2] and seq[1] < candSeq[2]:
                    if len(candSeq[0] - seq[0]) < support:
                        setRemove.append(j)
                    else:
                        setOverlap.append([j,candSeq[0] - seq[0]])
            if len(setRemove) > 1:
                del counter[c]
                candidates.pop(idx)
                del sequenceCluster[c]
                continue
            for _c in counter:
                if _c == c:
                    continue
                candSeq = sequenceCluster[_c]
                tmp = set()
                if candSeq[1] < seq[2] and seq[1] < candSeq[2]:
                    tmp = candSeq[0] & seq[0]
                counter[_c] -= len(tmp)
                sequenceCluster[_c][0] = sequenceCluster[_c][0] - tmp
            result.append(c)
            candidates.pop(idx)
            del counter[c]
            for j in setRemove:
                candidates.remove(j)
                del sequenceCluster[j]
                del counter[j]
            for j in setOverlap:
                sequenceCluster[j[0]][0] = j[1]
                counter[j[0]] = len(j[1])
    if len(result) == 0:
        return []
    return result 

def chunk_array(arr, k):
    """Divide arr into non-overlapping chunks of size k."""
    # trim the array to be divisible by k
    if len(arr) % k != 0:
        arr = arr[:-(len(arr)%k)]
    return [arr[i:i+k] for i in range(0, len(arr), k)]

def contiguous_partitions_chunks(chunks):
    """Generate all contiguous partitions of a list of chunks."""
    n = len(chunks)
    results = []

    def backtrack(start, path):
        if start == n:
            results.append(path)
            return
        for end in range(start + 1, n + 1):
            # Concatenate chunks[start:end] as a new group
            merged = np.concatenate(chunks[start:end])
            backtrack(end, path + [merged])
    
    backtrack(0, [])
    return results

class TiVy():
    def __init__(self,chunk_size=10,B=20,strength=1,max_iter=None,steps=0,nPerm=20,bandWidth=0.5,segment = False,cutoff=0,use_LSH=True,grid_search_spec=None,LSH_sample=1,random_seed=42):
        self.debug=False
        self.chunk_size=int(chunk_size)

        self.B=B
        self.step =steps
        self.strength=strength
        self.max_iter=max_iter
        self.nPerm = nPerm
        self.bandWidth = bandWidth
        self.segment = segment
        self.cutoff=cutoff
        self.use_LSH = use_LSH
        self.LSH_sample = LSH_sample
        self.do_grid_search = True if grid_search_spec is not None else False
        self.grid_search_spec = grid_search_spec
        self.random_seed = 42
    
    def cluster_subsequences(self,data):
        self.data=np.array(data)
        self.ts=np.array(data) #row columns at the same time
        self.chunks=math.floor(self.data.shape[1]/self.chunk_size)
        self.ts = self.ts[:,:self.chunks*self.chunk_size]
        self.ts=np.split(self.ts,self.chunks,axis=1)
        if self.ts[-1].shape[1]!=self.chunk_size:
            self.ts = self.ts[:-1]
        self.clusters=[]
        self.starts=[]
        self.ts_start=[]
        abs_max = np.max(np.abs(self.data.T))
        
        args=[]
        for ts in self.ts:
            args.append({
                'ts':ts/abs_max,
                'B':self.B,
                'max_iter':self.max_iter,
                'strength':self.strength,
                'step':self.step,
                'segment':self.segment,
                'cutoff': self.cutoff,
                'nPerm': self.nPerm,
                'bandWidth': self.bandWidth,
                'use_LSH': self.use_LSH,
                'LSH_sample': self.LSH_sample
            })  
        self.clusters =[create_clusters(arg) for arg in args]
                
        i = 0
        start = 0
        cumulative_time = 0  # Track cumulative time position

        for c in self.clusters:
            self.ts_start.append(cumulative_time)
            c[c!=0] += start
            c = c.astype(int)

            start = np.max(c)
            self.starts.append(start)
            
            # Use actual chunk size instead of fixed chunk_size
            actual_chunk_size = self.ts[i].shape[1]  # Dynamic chunk size
            cumulative_time += actual_chunk_size
            i += 1

        self.ts_start.append(cumulative_time)


    def get_size_of_cluster(self,seq):
        start = 0
        for s in self.starts:
            if seq[0]<=s:
                break
            start += 1
        end = 0
        for s in self.starts:
            if seq[-1]<=s:
                break
            end += 1  
        size = self.ts_start[end + 1] - self.ts_start[start]
        return size

    def get_timeseries_by_cluster(self,seq):
        return self.sequenceCluster[seq]

    def generate_counter(self,tree,seq,possibleCounts=[],counter=Counter()):
        for k in tree['children']:
            if seq == None:
                key = (int(k),)
            else:
                key = seq + (int(k),)
            counter[str(key)] = tree['children'][k]['count']
            possibleCounts.append(tree['children'][k]['count'])
            # self.sequenceCluster[str(key)][0]= idx
            self.generate_counter(tree['children'][k],key,possibleCounts,counter)
            
    def grid_search_clustering(self,data):
        self.data=np.array(data)
        self.ts=np.array(data) #row columns at the same time
        abs_max = np.max(np.abs(self.data.T))
        indices = np.arange(self.data.shape[1])
        chunks = chunk_array(indices, self.chunk_size)
        partitions_combinations = contiguous_partitions_chunks(chunks)
        cluster_caches = {}
        self.clusters_combinations = []
        print("start clustering")
        for partitions in tqdm(partitions_combinations):
            clusters = []
            time_ranges = [0]
            for p in partitions:
                h = (p[0],p[-1])
                if h not in cluster_caches:
                    ts = self.ts[:,h[0]:h[1]]
                    args = {
                        'ts':ts/abs_max,
                        'B':self.B,
                        'max_iter':self.max_iter,
                        'strength':self.strength,
                        'step':self.step,
                        'segment':self.segment,
                        'cutoff': self.cutoff,
                        'nPerm': self.nPerm,
                        'bandWidth': self.bandWidth,
                        'use_LSH': self.use_LSH,
                        'LSH_sample': self.LSH_sample
                    }
                    cluster_caches[h] = create_clusters(args)
                clusters.append(cluster_caches[h])
                time_ranges.append(h[1]+1)
            start_c_idx = 0
            for c in clusters:
                c += start_c_idx
                start_c_idx = max(c)
            self.clusters_combinations.append({'clusters':clusters,'time_range':time_ranges})
            
        print("end clustering")

    def grid_search_scanning(self):
        min_support = self.grid_search_spec['min_support']
        min_length = self.grid_search_spec['min_length']
        min_size = self.grid_search_spec['min_size']
        min_scan_support = self.grid_search_spec['min_scan_support']
        
        for i,clusters_combination in tqdm(enumerate(self.clusters_combinations)):
            clusters = clusters_combination['clusters']
            time_range = clusters_combination['time_range']
            scan_profile = self.prefix_scan(clusters,time_range,min_scan_support)
            counter = scan_profile['counter']
            sequenceClusterDB = scan_profile['sequenceClusterDB']
            result = self.optimize_patterns(sequenceClusterDB,counter,minSupport=min_support,minLength=min_length,minSize=min_size)
            
            clusters_combination['result'] = result
            
    def prefix_scan(self,clusters,timeIntervals,minlen=10,verbose=False):
        db=np.array(clusters).transpose()
        counter=Counter()
        tree = {'children':{}}
        sequenceCluster = {}
        possibleCounts=[]
        for i in range(db.shape[1]):
            indice = np.array(range(db.shape[0]))
            
            self.scan(sequenceCluster,timeIntervals,db[:,i:],None,minlen,tree['children'],indice,level=i,start=i,verbose=verbose)
        self.generate_counter(tree,None,possibleCounts,counter)
        return {'sequenceClusterDB':sequenceCluster,'possibleCounts':possibleCounts,'counter':counter} 
            
    # def prefix_scan(self,minlen=10,clusters=None):
    #     if clusters is None:
    #         clusters = self.clusters
    #     db=np.array(self.clusters).transpose()
    #     self.counter=Counter()
    #     self.tree = {'children':{}}
    #     self.sequenceCluster = {}
    #     self.possibleCounts=[]
    #     for i in range(db.shape[1]):
    #         indice = np.array(range(db.shape[0]))
    #         self.scan(self.sequenceCluster,self.ts_start,db[:,i:],None,minlen,self.tree['children'],indice,level=i,start=i)
    #     self.generate_counter(self.tree,None,self.possibleCounts,self.counter)     


    def scan(self,sequenceClusterDB,timeIntervals,db,seq,minlen,tree,indice,level=None,debug=None,start=0,hierarchy={},verbose=False):
        if verbose:
            print(seq,start)
        if db.size==0 or len(db)<minlen:
            return
        if db.ndim==1: 
            _db=db
        else:
            _db=db[:,0]
        unique, index, counts = np.unique(_db,return_counts=True,return_inverse=True)
        max_key = np.max(unique)
        for s,c in zip(unique, counts): #char,count
            if s==0:
                continue
            if level != None:
                if level not in hierarchy:
                    hierarchy[level] = []
                hierarchy[level].append((s,))
            if seq==None:
                key=(int(s),)
            else:
                key=seq+(int(s),)
            if c<minlen:
                continue
            tree[str(int(s))] = {'count':int(c),'children':{}}
            mask=np.where(index==np.where(unique==s)[0][0])[0]
            sequenceClusterDB[str(key)] = [set(list(map(int,indice[mask].tolist()))),int(timeIntervals[start]),int(timeIntervals[start+len(key)])]
            if db.shape[1]>=1:
                filterDb=db[mask,1:]
                self.scan(sequenceClusterDB,timeIntervals,filterDb,key,minlen,tree[str(s)]['children'],indice[mask],level=None,debug=c,start=start,hierarchy=hierarchy,verbose=verbose)
                
    def optimize_patterns(self,sequenceClusterDB,counter,minSupport=0,minLength=0,maxLength=None,minSize=0,maxSize=None,mask=None,labels=None):
        # counter= dict(self.counter)
        data = self.data
        sequenceCluster = {}
        for c in sequenceClusterDB:
            item = sequenceClusterDB[c]
            sequenceCluster[c] = []
            sequenceCluster[c].append(set(item[0]))
            sequenceCluster[c].append(item[1])
            sequenceCluster[c].append(item[2])
        stuff = optimize_sequence_by_support(support=minSupport,counter=counter,sequenceCluster=sequenceCluster)
        clusters = []
        if labels is not None:
            for _l in np.unique(labels):
                idx = set(np.where(labels == _l)[0])
                for c in stuff:
                    _idx = set(sequenceCluster[c][0]) & idx
                    clusters.append([_idx,sequenceCluster[c][1],sequenceCluster[c][2],_l])
        else:
            for c in stuff:
                clusters.append([list(sequenceCluster[c][0]),sequenceCluster[c][1],sequenceCluster[c][2]])
        result = []
        for c in clusters:
            length = c[2] - c[1]
            if mask != None:
                idx = list(set(c[0]) & mask)
            else:
                idx = list(c[0])
            size = len(idx)
            if length >= minLength and (maxLength == None or maxLength >= length) and (maxSize == None or maxSize >= size) and size > minSize:
                tmp = c
                tmp[0] = idx
                result.append(tmp)
                
        if labels is not None:
            results = {}
            medoid = {}
            for r in result:
                if r[3] not in results:
                    results[r[3]] = []
                    medoid[r[3]] = []
                results[r[3]].append(r)
                
                # get centroid
                subset = data[r[0],r[1]:r[2]]
                random_indice = np.random.choice(np.arange(subset.shape[0]), min(50,subset.shape[0]), replace=False)
                _centroid = subset[random_indice][get_average_sequence(subset[random_indice]),:]
                centroid = np.zeros(data.shape[1])
                centroid[r[1]:r[2]] = _centroid
                medoid[r[3]].append(centroid)
            result = []
            for l in results:
                medoid[l] = np.array(medoid[l])
                sorted_idx = get_flat_list(medoid[l])
                result += [results[l][i] for i in sorted_idx]
                
        return result

    def output_result(self,clusters,show_main_trend=True,return_idx=False):
        result = []
        output = {}
        output['labels'] = []
        if return_idx:
            output['idx'] = []
        if show_main_trend:
            output['center_line'] = []
        x_limit = self.data.shape[1]
        for c in clusters:
            if len(c) == 4:
                output['labels'].append(c[-1])
            ts = np.zeros((len(c[0]),x_limit))
            ts[:] = np.nan
            # print(c[1],c[2])
            subset = self.data[c[0],c[1]:c[2]]
            ts[:,c[1]:c[2]] = subset
            result.append(ts)
            if return_idx:
                output['idx'].append(c[0])
            if show_main_trend:
                _cl = np.zeros(x_limit)
                _cl[:] = np.nan
                random_indice = np.random.choice(np.arange(subset.shape[0]), min(50,subset.shape[0]), replace=False)
                centroid = subset[get_average_sequence(subset[random_indice]),:]
                centroid = np.trim_zeros(centroid)
                _cl[c[1]:centroid.shape[0]+c[1]] = centroid
                output['center_line'].append(_cl)
        output['result'] = result
        return output
                
    # def visualize(self,clusters,labels = None,n_columns=5,chart_width=5,chart_height=2,alpha=0.01,show_main_trend=True,dpi=300):
    #     color_codes = ['#6388b4', '#ffae34', '#ef6f6a', '#8cc2ca', '#55ad89', '#c3bc3f', '#bb7693', '#baa094', '#a9b5ae', '#767676']
    #     color = {}
    #     for i,l in enumerate(np.unique(labels)):
    #         color[l] = color_codes[i % len(color_codes)]
    #     print(color)
    #     result = []
    #     max_size = np.max([len(c[0]) for c in clusters])
    #     min_size = np.min([len(c[0]) for c in clusters])
    #     for c in clusters:
    #         if labels is not None:
    #             result.append({'data':c,'color':color[c[3]]})
    #         else:
    #             result.append({'data':c})
                
    #     n_chunks = self.chunks
    #     chunk_size = self.chunk_size
    #     n_chunks_per_column = n_columns * n_chunks
    #     data =  self.data 
        
    #     coord = []
    #     i = 0
    #     j = 0
    #     current_label = None
    #     for _r in result:
    #         r = _r['data']
    #         if labels is not None:
    #             if current_label == None:
    #                 current_label = r[3]
    #             if current_label != r[3]:
    #                 current_label = r[3]
    #                 j = 0
    #                 i += 1
    #         width = (r[2] - r[1]) // chunk_size
    #         if (j + width) > n_chunks_per_column:
    #             j = 0
    #             i += 1
    #         coord.append([i,j])
    #         j += width

    #     n_rows = i + 1
        
    #     fig = plt.figure(figsize=( chart_width * n_columns,n_rows * chart_height),dpi=dpi)
    #     gs = fig.add_gridspec(n_rows, n_chunks_per_column,wspace=10, hspace=0.4)
    #     fig.tight_layout()

    #     for i,_r in tqdm(enumerate(result)):
    #         r = _r['data']
    #         if labels is not None:
    #             color = _r['color']
    #         else:
    #             color = '#6388b4'
    #         width = (r[2] - r[1]) // chunk_size
    #         subset = data[r[0],r[1]:r[2]]
    #         x = np.arange(r[1],r[2])
    #         axs = fig.add_subplot(gs[coord[i][0], coord[i][1]: coord[i][1] + width])
    #         axs.set_xlim(r[1],r[2])
    #         axs.set_ylim(np.min(self.data),np.max(self.data))
    #         axs.set_xticks([x[0],x[-1]])
    #         axs.set_xticklabels([r[1],r[2]])
    #         axs.set_yticklabels([])
    #         axs.tick_params(axis='x', labelsize= 3)
            
    #         random_indice = np.random.choice(np.arange(subset.shape[0]), min(50,subset.shape[0]), replace=False)
    #         centroid = subset[random_indice][get_average_sequence(subset[random_indice]),:]
    #         centroid = np.trim_zeros(centroid)
    #         # print(centroid)
    #         if show_main_trend:
    #             upper_bound = np.max(subset,axis=0)
    #             lower_bound = np.min(subset,axis=0)
    #             axs.fill_between(x,upper_bound,lower_bound,alpha=0.3,color=color)
    #             axs.plot(centroid,color=color)
    #         else:
    #             lines = []
    #             for y in subset:
    #                 _y = np.trim_zeros(y,'b')
    #                 _x = x[:_y.shape[0]]
    #                 lines.append(np.column_stack([_x,_y]))
    #             line_segments = LineCollection(lines,color=color,alpha=alpha)
    #             axs.add_collection(line_segments)
    #             scale = 5 * (subset.shape[0] - min_size) / (max_size - min_size) + 1 *(1 - (subset.shape[0] - min_size) / (max_size - min_size))
    #             # print(scale)
    #             # axs.plot(centroid,color='black',linewidth=scale)

    #     plt.show()
    