import numpy as np
from .util import *
from shapely.geometry import LineString
from .geometry import polygon2mesh

def using_clump(a):
    return np.ma.clump_unmasked(np.ma.masked_invalid(a))

def _make_line_mesh_buffer(d_x,d_y,size):
    masks = using_clump(d_y)
    line = np.vstack((d_x,d_y)).T
    lines = [line[m,:] for m in masks]
    mesh_vertice = []
    mesh_face = []
    last_index = 0
    for l in lines:
        if l.shape[0] < 2: continue
        line = LineString(l).buffer(size).exterior.coords.xy
        polygon = np.vstack((line[0],line[1])).T.tolist()
        mv, mf = polygon2mesh(polygon)
        mf += last_index
        mesh_vertice.append(mv)
        mesh_face.append(mf)
        last_index = mf[-1,-1] + 1
    return np.concatenate(mesh_vertice), np.concatenate(mesh_face)

def _make_line_density_buffer(_d_x,_d_y,h_range,n_bins_x,n_bins_y):
    d_x = np.linspace(0,np.max(_d_x),n_bins_x*2)
    d_y = [np.interp(d_x,_d_x,y) for y in _d_y]
    d_y = np.array(d_y)
    n_rows, n_cols = d_y.shape
    x = np.tile(d_x,n_rows)
    y = d_y.reshape(n_rows * n_cols)
    H,xedges,yedges = np.histogram2d(x,y,bins=[n_bins_x,n_bins_y],range=h_range)
    xv,yv = np.meshgrid(np.linspace(0,1,n_bins_x+1),np.linspace(0,1,n_bins_y+1))
    grid = np.dstack((xv,yv))
    p1 = grid[:-1,:-1].reshape(-1,2)
    idx1 = np.arange(p1.shape[0])
    p2 = grid[:-1,1:].reshape(-1,2)
    idx2 = np.arange(p2.shape[0]) + np.max(idx1) + 1
    p3 = grid[1:,1:].reshape(-1,2)
    idx3 = np.arange(p3.shape[0]) + np.max(idx2) + 1
    p4 = grid[1:,:-1].reshape(-1,2)
    idx4 = np.arange(p4.shape[0]) + np.max(idx3) + 1

    f1 = np.dstack((idx1,idx2,idx3))[0]
    f2 = np.dstack((idx3,idx4,idx1))[0]
    mesh_face = np.concatenate((f1,f2))
    mesh_vertices = np.concatenate((p1,p2,p3,p4))
    mesh_vertices = np.hstack((mesh_vertices,np.zeros(mesh_vertices.shape[0]).reshape(-1,1)))
    mesh_alpha = np.tile(H.T.reshape(-1),4)
    return mesh_vertices, mesh_face, mesh_alpha

def _make_line_buffer(d_x,d_y):
    n_rows, n_cols = d_y.shape
    d_x_tile = np.tile(d_x,(n_rows,1))
    v = np.dstack((d_x_tile,d_y,np.ones((n_rows, n_cols))))

    non_empty_mask = np.logical_not(np.isnan(d_y))
    # v = v[non_empty_mask]
    v[~non_empty_mask]= [0,0,0]
    v = v.reshape(v.shape[0]*v.shape[1],3)
    #set indices
    indices = np.arange(n_rows*n_cols).reshape((n_rows,n_cols))
    indices_next = indices[:,1:]
    indices_current = indices[:,:-1]
    mask_next = non_empty_mask[:,1:]
    mask_current = non_empty_mask[:,:-1]
    s = np.dstack((indices_current,indices_next))[mask_current & mask_next]
    # v_offset = s[-1][1] + 1
    return v,s

def _make_area_buffer(d_x,d_y,q=1):
    n_rows, n_cols = d_y.shape
    with np.errstate(divide='ignore'):
        y_min = np.nanquantile(d_y,1-q,0)
        y_max = np.nanquantile(d_y,q,0)
    # mean = np.nanmedian(d_y)
    # y_min[np.isnan(y_min)] = mean
    # y_max[np.isnan(y_max)] = mean
    masks = using_clump(y_min)
    # polygon = np.concatenate((np.dstack((d_x,y_max))[0],np.dstack((d_x[::-1],y_min[::-1]))[0]))
    mesh_vertice = []
    mesh_face = []
    last_index = 0
    for m in masks:
        _max = np.vstack((d_x[m],y_max[m])).T
        _min = np.vstack((d_x[m][::-1],y_min[m][::-1])).T
        p = np.concatenate((_max,_min)).tolist()
        mv, mf = polygon2mesh(p)
        if len(mf) == 0:
            continue
        mf += last_index
        mesh_vertice.append(mv)
        mesh_face.append(mf)
        last_index = mf[-1,-1] + 1
    if len(mesh_vertice) == 0:
        return None, None
    return np.concatenate(mesh_vertice), np.concatenate(mesh_face)


def _make_frame_buffer(vertices,colors,connect,v_offset,texts, \
                       x_offset,y_offset,width_chart, height_chart, padding_width_scaled, padding_height_scaled, x_ticks, y_ticks,n_ticks):
    #draw frame
    #set vertice locations
    x1 = x_offset
    x2 = width_chart + x_offset
    y1 = y_offset
    y2 = height_chart + y_offset
    frame_vertice = np.array([[x1,y1,1],[x2,y1,1],[x2,y2,1],[x1,y2,1]])
    #set vertice color
    frame_c = np.tile(FRAME_COLOR,(4,1))
    #append
    if vertices is None:
        vertices = frame_vertice
    else:
        vertices = np.concatenate((vertices,frame_vertice))
    if colors is None:
        colors = frame_c
    else:
        colors = np.concatenate((colors,frame_c))
    #set lines
    s1 = np.arange(4)[:-1]
    s1 += v_offset
    s2 = s1 + 1
    s = np.column_stack((s1,s2))
    s = np.concatenate((s,[[s1[0],s2[-1]]]))
    if connect is None:
        connect = s
    else:
        connect = np.concatenate((connect,s))
    v_offset = s2[-1] + 1
    #draw y ticks 
    # labels
    tick_padding = padding_width_scaled*0.2
    texts['text'] += y_ticks
    pos = np.linspace(y1,y2,n_ticks).tolist()
    pos = [[x1-tick_padding,p] for p in pos]
    texts['pos'] += pos
    # ticks vertice
    tick_vertice = [[p[0]+tick_padding/2,p[1],1] for p in pos] + [[p[0]+tick_padding,p[1],1] for p in pos]
    tick_c = np.tile(FRAME_COLOR,(len(tick_vertice),1))
    vertices = np.concatenate((vertices,tick_vertice))
    colors = np.concatenate((colors,tick_c))
    # tick lines
    s = [[v_offset+i,v_offset+len(pos)+i] for i in range(len(pos))]
    connect = np.concatenate((connect,s))
    v_offset = s[-1][1] + 1
    #draw x ticks
    # labels
    tick_padding = padding_height_scaled*0.2
    texts['text'] += x_ticks
    pos = np.linspace(x1,x2,len(x_ticks)).tolist()
    pos = [[p,y1-tick_padding] for p in pos]
    texts['pos'] += pos
    # ticks vertice
    tick_vertice = [[p[0],p[1]+tick_padding/2,1] for p in pos] + [[p[0],p[1]+tick_padding,1] for p in pos]
    tick_c = np.tile(FRAME_COLOR,(len(tick_vertice),1))
    vertices = np.concatenate((vertices,tick_vertice))
    colors = np.concatenate((colors,tick_c))
    # tick lines
    s = [[v_offset+i,v_offset+len(pos)+i] for i in range(len(pos))]
    connect = np.concatenate((connect,s))
    v_offset = s[-1][1] + 1

    return vertices,colors,connect,v_offset,texts