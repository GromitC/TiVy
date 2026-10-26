import numpy as np
import vispy
import vispy.scene
from vispy import app
from vispy.scene import visuals
from operator import itemgetter
import math
from vispy.scene.cameras import PanZoomCamera
from vispy.geometry import Rect
import ipywidgets as widgets
from .buffer import _make_line_buffer, _make_area_buffer,_make_frame_buffer, _make_line_mesh_buffer,_make_line_density_buffer
from .util import *
from .geometry import polygon2mesh
from tqdm.notebook import tqdm
from collections import defaultdict

'''
todos:
1. add central line
2. facet
3. filters
4. area chart
'''

#vispy bug: text cannot be zoomed; quick fix on the zoom function
class CustomPanZoomCamera(PanZoomCamera):
    def __init__(self, texts, **kwds):
            self.texts = texts
            self.font_sizes = [12 for i in range(len(texts))]
            self.currect_scale = 1
            super().__init__(**kwds)
    
    def set_font_size(self,font_sizes):
        self.font_sizes = font_sizes
        self.font_size_initialized = True
        
    def zoom(self, factor, center=None):
        # Init some variables
        center = center if (center is not None) else self.center
        assert len(center) in (2, 3, 4)
        # Get scale factor, take scale ratio into account
        if np.isscalar(factor):
            scale = [factor, factor]
        else:
            if len(factor) != 2:
                raise TypeError("factor must be scalar or length-2 sequence.")
            scale = list(factor)
        if self.aspect is not None:
            scale[0] = scale[1]
        # Make a new object (copy), so that allocation will
        # trigger view_changed:
        rect = Rect(self.rect)
        # Get space from given center to edges
        left_space = center[0] - rect.left
        right_space = rect.right - center[0]
        bottom_space = center[1] - rect.bottom
        top_space = rect.top - center[1]
        # Scale these spaces
        rect.left = center[0] - left_space * scale[0]
        rect.right = center[0] + right_space * scale[0]
        rect.bottom = center[1] - bottom_space * scale[1]
        rect.top = center[1] + top_space * scale[1]
        self.rect = rect
        self.currect_scale *= factor
        for text, font_size in zip(self.texts,self.font_sizes):
            text.font_size = font_size / self.currect_scale
        

class TimeSeriesVis():
    
    def __init__(self,x_domain = None,chart_width=300,chart_height=150,padding=50,n_cols=3,n_ticks=5,interpolate=200,verbose=0):
        self.chart_width = chart_width
        self.chart_height = chart_height
        self.padding = padding
        self.n_cols = n_cols
        self.n_ticks = n_ticks
        self.camera = None
        self.line = None
        self.x_domain = x_domain
        self.opacity = 0.3
        self.interpolate = interpolate
        self.line_central = [] 
        self.area = []
        self.verbose = verbose
        
    def fit(self,data,labels=None,labels_order=None,central_line=None,area=False):
        if labels is None:
            self.make_labels = False
        else:
            self.make_labels = True
        if central_line is None:
            self.has_central_line = False
        else:
            self.has_central_line = True
        self.size_threshold = 0
        self.data = [np.array(d) for d in data]
        self.interval_threshold = [0,self.data[0].shape[1]]
        self.data,labels,central_line = self.make_canvas_spec(self.data,labels,labels_order,central_line)
        self.central_line = central_line
        self.labels = labels
        # self.init_canvas()
        self.prepare_buffers(central_line)
        # self.paint()
        
    def visualize(self):
        self.make_buttons(self.central_line)
        self.init_canvas()
        self.paint()
        return self.canvas
        
    def paint(self):
        if self.verbose == 1:
            print('refresh') 
            
        vertices, colors, connect, \
        vertices_util, colors_util, connect_util, \
        texts, label_texts, \
        vertices_central, face_central, \
        vertices_area, colors_area, faces_area \
        = self.make_buffer(self.buffers)
        
        self.draw(vertices, colors, connect, \
                    vertices_util, colors_util, connect_util, \
                    texts, label_texts, \
                    vertices_central, face_central, \
                     vertices_area, colors_area, faces_area)
        
    def make_buttons(self,central_line=None):
        self.show_area = True
        if central_line is None:
            self.show_central_line = False
        else:
            self.show_central_line = True
        if not self.make_labels:
            self.sort_labels = False
        else:
            self.sort_labels = True
        
        self.counter = 1
        def change_checkbox_area(e):
            if e['name'] == 'value':
                if self.counter == 1: #weird bug of not having the correct value first time
                    self.show_area = not e['new']
                    self.counter += 1
                else:
                    self.show_area = not e['new']
                self.paint()
                
        def change_checkbox_central_line(e):
            if e['name'] == 'value':
                self.show_central_line = e['new']
                self.paint()
        
        def change_checkbox_labels(e):
            if e['name'] == 'value':
                self.make_labels = e['new']
                self.paint()
            
        def change_size_slider(e):
            if e['name'] == 'value':
                self.size_threshold = e['new']
                self.paint()
                            
        def change_inteval_slider(e):
            if e['name'] == 'value':
                self.interval_threshold = e['new']
                self.paint()

        def change_opacity_slider(e):
            if e['name'] == 'value':
                self.opacity = e['new']
                self.paint()

        items_layout = widgets.Layout( width='auto')
            
        buttons = []
        interactions = []

        label = widgets.Label('Show Lines')
        buttons.append(label)
        interactions.append(None)

        checkbox_area = widgets.Checkbox(
            value=False,
            description='|',
            disabled=False,
            indent=False,
            layout=items_layout
        )
        buttons.append(checkbox_area)
        interactions.append(change_checkbox_area)
        
        if central_line is not None:
            label = widgets.Label('Show Central Line')
            buttons.append(label)
            interactions.append(None)

            checkbox_central_line = widgets.Checkbox(
                value=True,
                description='|',
                disabled=False,
                indent=False,
                layout=items_layout
            )
            buttons.append(checkbox_central_line)
            interactions.append(change_checkbox_central_line)
        
        if self.make_labels:
            label = widgets.Label('Sort By Labels')
            buttons.append(label)
            interactions.append(None)

            checkbox_sort_labels = widgets.Checkbox(
                value=True,
                description='',
                disabled=False,
                indent=False,
                layout=items_layout
            )
            buttons.append(checkbox_sort_labels)
            interactions.append(change_checkbox_labels)
        
        label = widgets.Label('| Opacity: ')
        buttons.append(label)
        interactions.append(None)

        opacity_slider = widgets.FloatSlider(
            min=0,
            max=1,
            value=self.opacity,
            step=0.01,
            continuous_update=False,
            readout_format='.2f',
            layout=widgets.Layout( width='10%')
        )
        buttons.append(opacity_slider)
        interactions.append(change_opacity_slider)

        size_max = np.max([d.shape[0] for d in self.data])
        
        label = widgets.Label('| Filter By Number of Series: ')
        buttons.append(label)
        interactions.append(None)

        size_slider = widgets.IntSlider(
            min=0, 
            max=size_max+1, 
            step=1, 
            value=0,
            continuous_update=False,
            layout=widgets.Layout( width='10%')
        )
        
        buttons.append(size_slider)
        interactions.append(change_size_slider)

        label = widgets.Label('| Filter By Time Intervals: ')
        buttons.append(label)
        interactions.append(None)

        interval_slider = widgets.IntRangeSlider(
            value=self.interval_threshold,
            min=self.interval_threshold[0],
            max=self.interval_threshold[1],
            step=1,
            # description='| Intervals:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout=widgets.Layout( width='10%')
        )
        buttons.append(interval_slider)
        interactions.append(change_inteval_slider)

        box_layout = widgets.Layout(display='flex',
                    flex_flow='row',
                    align_items='stretch',
                    border='solid',
                    width='100%')

        box = widgets.Box(children=buttons, layout=box_layout)
        
        display(box)
        
        for b,i in zip(buttons,interactions):
            if i is not None:
                b.observe(i)
        
        
    def make_canvas_spec(self,data,labels=None,labels_order=None,central_line=None):
        if self.verbose == 1:
            print('make_canvas_spec')
            
        self.meta = []
        #check sizes of list of list match
        interval = set([l.shape[1] for l in data])
        if len(interval) > 1:
             raise ValueError('not all lists have same length!')
                
        n_charts = len(data)
        
        self.x_limit = data[0].shape[1]
        self.y_limit = (np.min([np.nanmin(l) for l in data]),np.max([np.nanmax(l) for l in data]))
        
        if labels is None:
            self.n_rows = math.ceil(n_charts/self.n_cols)
            data_sort = data
        else:
            
            if labels_order is None:
                idx_ = defaultdict(list) 
                for i,l in enumerate(labels):
                    idx_[l].append(i)
                idx = []
                for k in idx_:
                    idx += idx_[k]
            else:
                order = {k:i for k,i in zip(labels_order,list(range(len(labels_order))))}
                idx = [order[l] for l in labels]
                idx_ = defaultdict(list)
                for i,l in enumerate(idx):
                    idx_[l].append(i)
                idx = []
                for i in range(len(labels_order)):
                    idx += idx_[i]
            
            data_sort = []
            for i in idx:
                data_sort.append(data[i])
                self.meta.append({'id':str(i),'count':str(data[i].shape[0])})
            if central_line is not None:
                central_line = [central_line[i] for i in idx]
            labels = [labels[i] for i in idx]
            (unique, counts) = np.unique(labels, return_counts=True)
            self.n_rows = sum([math.ceil(c/self.n_cols) for c in counts])
            self.color = {k:tableau10[i%len(tableau10)] for k,i in zip(unique,list(range(len(unique))))}
            self.colorLight = {k:tableau10light[i%len(tableau10light)] for k,i in zip(unique,list(range(len(unique))))}
            
        n_rows = self.n_rows
        n_cols = self.n_cols
        chart_width = self.chart_width
        chart_height = self.chart_height
        padding = self.padding
        # chart_width *= 96 #inch to pixels
        # chart_height *= 96
        
        self.canvas_width = (chart_width + padding) * n_cols
        self.canvas_height = (chart_height + padding) * n_rows

        return data_sort,labels,central_line
        
    def init_canvas(self):
        if self.verbose == 1:
            print('init_canvas')
        app.use_app('jupyter_rfb')
        ca = vispy.scene.SceneCanvas(keys='interactive', show=True,size=(self.canvas_width,self.canvas_height),bgcolor=[1,1,1])
        ca.measure_fps()
        ca.show()
        self.canvas = ca
        view = ca.central_widget.add_view()
        
        self.text = visuals.Text(anchor_x='right')
        view.add(self.text)
        self.label_texts = visuals.Text(anchor_x='right')
        view.add(self.label_texts)
               
        self.camera = CustomPanZoomCamera([self.text,self.label_texts])
        view.camera = self.camera

        axis = visuals.Axis(parent=view.scene)
        
        self.view = view
        
        self.line = visuals.Line()
        self.line.set_data(pos=np.array([[0,0,0]]),color=np.array([[0,0,0,0]]))
        self.line.order = 1
        self.view.add(self.line)
        
        self.mesh = visuals.Mesh()
        self.mesh.order = 0 
        self.view.add(self.mesh)
        
        app.run()
                
    def prepare_buffers(self,central_line=None):
        data = self.data
        self.set_widthScale(self.data)
        if self.verbose == 1:
            print('prepare_buffer')
        self.buffers = []
        x_limit = self.x_limit
        d_x = np.arange(self.x_limit)
        central_line_idx = 0
        print('Preparing Buffer...')
        for d in tqdm(data):
            buffers = { 'vertices':None,'connect':None, 'vertices_central':None,'connect_central':None,'v_offset_central':0 }
            # buffers['vertices'],buffers['connect'] = _make_line_buffer(d_x,d)
            h_range = [[0,x_limit],self.y_limit]
            aspect_ratio =  self.chart_height/self.chart_width
            n_bins_x = max(self.interpolate,self.x_limit)
            n_bins_y = int(aspect_ratio * n_bins_x)
            buffers['vertices_line'], buffers['faces_line'],  buffers['alpha_line']= _make_line_density_buffer(d_x,d,h_range, n_bins_x, n_bins_y)
            mv1,mf1 = _make_area_buffer(d_x,d,0.9)
            mv2,mf2 = _make_area_buffer(d_x,d,1)
            mask = ~np.all(np.isnan(d),0)
            interval = np.where(mask)[0][[0,-1]]
            buffers['interval'] = interval
            buffers['vertices_area'] = [mv1,mv2]
            buffers['face_area'] = [mf1,mf2]
            if central_line is not None:
                s = d.shape[0]
                mesh_vertice, mesh_face = _make_line_mesh_buffer(d_x/self.x_limit,central_line[central_line_idx]/self.y_limit[1],self.scale(s))
                buffers['vertices_central'] = mesh_vertice
                buffers['face_central'] = mesh_face
                central_line_idx += 1
            buffers['size'] = d.shape[0]
            self.buffers.append(buffers)
            
    def make_buffer(self,buffers):
        if self.verbose == 1: 
            print('make_buffer')
        padding = self.padding
        #scales
        x_limit, y_limit = self.x_limit, self.y_limit
        chart_width = self.chart_width #* 96 # inch to pixels
        chart_height = self.chart_height #* 96
        n_rows,n_cols = self.n_rows,self.n_cols
        #canvas
        canvas_width = (chart_width + padding) * n_cols
        canvas_height = (chart_height + padding) * n_rows
        padding_width_scaled = padding / canvas_width
        padding_height_scaled = padding / canvas_height
        width_chart = (1-MARGIN_X) / n_cols - padding_width_scaled
        height_chart = (1-MARGIN_Y) / n_rows - padding_height_scaled
        x_offset = padding_width_scaled
        y_offset = (1-MARGIN_Y) - height_chart 
        #ticks
        y_ticks = np.around(np.linspace(y_limit[0],y_limit[1],self.n_ticks),decimals=2).astype(str).tolist()
        if self.n_ticks == 1:
            x_ticks = np.array([self.x_limit])
        elif self.n_ticks <= 0:
            x_ticks = []
        else:
            x_ticks = np.linspace(0,self.x_limit,int((self.x_limit+1)/math.floor((self.x_limit+1)/self.n_ticks))).astype(int)
        if self.x_domain is not None:
            x_ticks = np.array(self.x_domain)[x_ticks[:-1]].astype(str).tolist()
        else:
            x_ticks = x_ticks.astype(str).tolist()
        #offset
        v_offset = 0
        v_offset_central = 0
        v_offset_util = 0
        #buffers
        # connect, vertices, colors = None, None, None
        faces,vertices,colors = [],None,[]
        vertices_util, colors_util, connect_util = None, None, None
        vertices_central,colors_central,connect_central = [], None, None
        face_central = []
        vertices_area, colors_area, faces_area = [], [], []
        #fonts
        texts = {'text':[],'color':FRAME_COLOR,'pos':[],'font_size':padding / 10}
        self.camera.set_font_size([texts['font_size'],self.chart_height / 5])
        #start iterations
        r = 1
        c = 1
        label_texts = None
        start_labels = True
        new_line = True
        if self.make_labels:
            color_idx = 0
            prev_label = None
            label_texts = []
        if self.has_central_line and self.show_central_line:
            central_line_idx = 0
#             self.set_widthScale(self.data,height_chart)
        for b in buffers:
            if b['size'] < self.size_threshold or \
                (b['interval'][0] > self.interval_threshold[1] or b['interval'][1] < self.interval_threshold[0]):
                if self.make_labels: 
                    color_idx += 1
                continue
            if self.make_labels:
                if prev_label != self.labels[color_idx]:
                    prev_label = self.labels[color_idx]
                    if not new_line:
                        c = 1
                        r += 1
                    x_offset = padding_width_scaled
                    y_offset = (1-MARGIN_Y) - height_chart * r - padding_height_scaled * (r-1)
                    label_texts.append({'text':self.labels[color_idx],'y':height_chart /2 + y_offset})
                    # continue
                prev_label = self.labels[color_idx]
                if not self.show_area:
                    color = self.color[self.labels[color_idx]]
                else:
                    color = self.colorLight[self.labels[color_idx]]
                color_idx += 1
            else:
                color = BLUE_COLOR
            if not self.show_area:
                #prepare lines
                #density lines
                #vertices
                if vertices is None:
                    vertices = []
                v = b['vertices_line'].copy()
                v[:,0] = v[:,0] * width_chart + x_offset
                v[:,1] = v[:,1] * height_chart + y_offset
                vertices.append(v)
                faces.append(b['faces_line'])
                alpha = b['alpha_line'] * self.opacity
                alpha[alpha>1] = 1
                color = np.tile(color,(v.shape[0],1))
                color = np.concatenate((color,alpha.reshape(-1,1)),1)
                colors.append(color)
                # line chart
                # v = b['vertices'].copy()
                # v[:,0] = v[:,0] / (x_limit-1) * width_chart + x_offset
                # v[:,1] = (v[:,1] - y_limit[0]) / (y_limit[1] - y_limit[0]) * height_chart + y_offset
                # if vertices is None:
                #     vertices = v
                # else:
                #     vertices = np.concatenate((vertices,v))  
                # # connect
                # _connect = b['connect'] + v_offset
                # if connect is None:
                #     connect = _connect
                # else:
                #     connect = np.concatenate((connect,_connect))
                # v_offset = vertices.shape[0]#_connect[-1][1] + 1
                # # color
                # color = np.append(color,self.opacity)
                # color = np.tile(color,(v.shape[0],1))
                # if colors is None:
                #     colors = color
                # else:
                #     colors = np.concatenate((colors,color))
            else:
                #vertice
                tmp = []
                if b['vertices_area'][0] is not None:
                    for v in b['vertices_area']:
                        _v = v.copy()
                        _v[:,0] = _v[:,0] / (x_limit-1) * width_chart + x_offset
                        _v[:,1] = (_v[:,1] - y_limit[0]) / (y_limit[1] - y_limit[0]) * height_chart + y_offset
                        tmp.append(_v)
                    vertices_area.append(tmp)
                    #color
                    colors_area.append(color)
                    #faces
                    faces_area.append(b['face_area'])
            #prepare central lines
            if self.has_central_line and self.show_central_line:
                #vertices
                cl = b['vertices_central'].copy()
                cl[:,0] *= x_limit
                cl[:,1] *= y_limit[1]
                cl[:,0] = cl[:,0] / (x_limit-1) * width_chart + x_offset
                cl[:,1] = (cl[:,1] - y_limit[0]) / (y_limit[1] - y_limit[0]) * height_chart + y_offset
                vertices_central.append(cl)
                face_central.append(b['face_central'])
            #frames
            vertices_util, colors_util, connect_util, v_offset_util, texts \
            = _make_frame_buffer(vertices_util, colors_util, connect_util, v_offset_util, texts , \
                            x_offset, y_offset, width_chart, height_chart, padding_width_scaled, padding_height_scaled, x_ticks, y_ticks,self.n_ticks)
            #chart locations
            if c != n_cols:
                x_offset += width_chart + padding_width_scaled
                c += 1
                new_line = False
            else:
                c = 1
                r += 1
                new_line = True
                x_offset = padding_width_scaled
            y_offset = (1-MARGIN_Y) - height_chart * r - padding_height_scaled * (r-1)
        #done
        # return vertices, colors, connect, \
        return vertices, colors, faces, \
                vertices_util, colors_util, connect_util, \
                texts, label_texts, \
                vertices_central, face_central, \
                vertices_area, colors_area, faces_area
    
    
    # def draw(self,vertices, colors, connect, \
    def draw(self,vertices, colors, faces, \
                    vertices_util, colors_util, connect_util, \
                    texts, label_texts, \
                    vertices_central, face_central,
                     vertices_area, colors_area, faces_area):#, \
        if self.verbose == 1:
            print('draw')
#         self.clear_line_central()
        self.clear_label_texts()
        #for area and central lines
        mesh_vertices = []
        mesh_faces = []
        mesh_vertices_color = []
        mesh_vertices_id = 0
        if vertices_central is not None and self.show_central_line:
            #make mesh lines
            for v, f in zip(vertices_central,face_central):
                color = [0,0,0,1]
                mesh_vertices.append(v)
                mf = f + mesh_vertices_id
                mesh_faces.append(mf)
                mesh_vertices_id = mf[-1,-1] + 1
                mesh_vertices_color.append(np.tile(color,(v.shape[0],1)))
           
        if label_texts is not None:
            self.label_texts.text = [t['text'] for t in label_texts]
            self.label_texts.pos = [[0,t['y']] for t in label_texts]
        if vertices is not None:
            for v, f, c in zip(vertices, faces, colors):
                mesh_vertices.append(v)
                mf = f + mesh_vertices_id
                mesh_faces.append(mf)
                mesh_vertices_id +=  np.max(f) + 1
                mesh_vertices_color.append(c)
            self.line.set_data(vertices_util,color=colors_util,connect=connect_util)
            # last_idx = vertices.shape[0]
            # connect_util += last_idx
            # connect_all = np.concatenate((connect,connect_util))
            # vertices_all = np.concatenate((vertices,vertices_util))
            # colors_all = np.concatenate((colors,colors_util))
            # self.line.set_data(vertices_all, color=colors_all,connect=connect_all)
        else:
            #draw area
            for vs, fs, c in zip(vertices_area, faces_area, colors_area):
                color = [[c[0],c[1],c[2],1],[0.9,0.9,0.9,1]]
                for v,f,c in zip(vs,fs,color):
                    mesh_vertices.append(v)
                    mf = f + mesh_vertices_id
                    mesh_faces.append(mf)
                    mesh_vertices_id = mf[-1,-1] + 1
                    mesh_vertices_color.append(np.tile(c,(v.shape[0],1)))
                # mesh_vertices_color.append(np.tile([0.9,0.9,0.9],(v.shape[0],1)))
            self.line.set_data(vertices_util,color=colors_util,connect=connect_util)
        #draw meshs
        if len(mesh_vertices) > 0:
            self.mesh.set_data(vertices=np.concatenate(mesh_vertices), faces=np.concatenate(mesh_faces), vertex_colors=np.concatenate(mesh_vertices_color))
        else:
            self.mesh.set_data(vertices=np.array([[0,0,0],[0,0,0],[0,0,0]]), faces=np.array([[0,1,2]]), vertex_colors=np.tile([0,0,0,0],(3,1)))
        #draw text
        if len(texts['text']) == 0:
            self.text.text = []
        else:
            self.text.text = texts['text']
            self.text.pos = texts['pos']
            self.text.font_size = texts['font_size'] /  self.camera.currect_scale
            self.text.order = 2
        
    def set_widthScale(self,data):
        extent = np.max([np.nanmax(d) for d in data]) - np.min([np.nanmin(d) for d in data])
        max_height = 0.015
        min_height = 0.005
        length = [len(d) for d in data]
        max_n_line = max(length) # a
        min_n_line = min(length)
        if min_n_line == max_n_line:
            self.scale = lambda x: 0.003
        else:
            a = np.array([[max_n_line,1],[min_n_line,1]])
            b = np.array([max_height,min_height])
            _x = np.linalg.solve(a, b)
            self.scale = lambda x: _x[0]*x + _x[1]

    def clear_label_texts(self):
        self.label_texts.text = []
    

    
        