def separateData(data,labels,center_line=None):
    data_group = []
    labels_group = []
    center_lines = []
    list_idx = []
    if center_line is None:
        iterator = zip(data,labels)
        center_lines = None
    else:
        iterator = zip(data,labels,center_line)
    idx = 0
    for it in iterator:
        if center_line is None:
            d,l = np.array(it[0]),np.array(it[1])
        else:
            d,l, cl = np.array(it[0]),np.array(it[1]),it[2]
        unique = np.unique(l)
        for u in unique:
            _d = d[l==u,:]
            if len(_d) > 10:
                data_group.append(d[l==u,:])
                list_idx.append(idx)
                labels_group.append(u)
            if center_line is not None:
                center_lines.append(cl)
        idx += 1
    return data_group, labels_group, center_lines, list_idx
    
def hex2rgb(color):
    h = color.lstrip('#')
    return [int(h[i:i+2], 16)/ 255.0 for i in (0, 2, 4)]

FRAME_COLOR = [0.5,0.5,0.5,1]
BLUE_COLOR = hex2rgb('#4E79A7')
BLACK_COLOR = hex2rgb('#000000')
MARGIN_X = 0.02
MARGIN_Y = 0.05
tableau10 = ["#4e79a7","#F28E2B","#E15759","#76B7B2","#59A14F","#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"]
tableau10 = [hex2rgb(t) for t in tableau10]
tableau10light = ['#AEC7E8','#FFBB78','#ff9896','#9edae5','#98DF8A','#dbdb8d','#c5b0d5','#f7b6d2','#c49c94','#c7c7c7']
tableau10light = [hex2rgb(t) for t in tableau10light]