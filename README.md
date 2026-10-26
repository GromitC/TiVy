# TiVy (Time Series Visual Summary)
Scalable Time Series Summarization Visualization in Jupyter Notebook
![](time-series-demo.gif)

## Installation
 1. Follow carefully the installation guide to install [vispy](https://vispy.org/installation.html) with its Jupyter extension (refer to the [notebook](https://vispy.org/installation.html#jupyter-notebook-extension) or [JupyterLab](https://vispy.org/installation.html#jupyterlab) sections).
 2. Check whether you can run vispy successfully by running the following code in Jupyter:
 ```
from vispy import scene
from vispy.visuals.transforms import STTransform

canvas = scene.SceneCanvas(
    keys="interactive", bgcolor="white", size=(500, 400), show=True, resizable=True
)

view = canvas.central_widget.add_view()
view.camera = "arcball"

sphere1 = scene.visuals.Sphere(
    radius=1, method="latitude", parent=view.scene, edge_color="black"
)

sphere2 = scene.visuals.Sphere(
    radius=1, method="ico", parent=view.scene, edge_color="black"
)

sphere3 = scene.visuals.Sphere(
    radius=1,
    rows=10,
    cols=10,
    depth=10,
    method="cube",
    parent=view.scene,
    edge_color="black",
)

sphere1.transform = STTransform(translate=[-2.5, 0, 0])
sphere3.transform = STTransform(translate=[2.5, 0, 0])

view.camera.set_range(x=[-3, 3])
canvas
 ```

 3. Install the remaining dependencies: ```pip install -r requirements.txt```

    TiVy calls dtaidistance's C routine (`dtw.distance_matrix_fast`). On recent Pythons the PyPI wheel has no C library, and a default (isolated) source build compiles against a newer numpy than the pinned one and fails at runtime with `module 'dtaidistance' has no attribute 'DTWSeriesMatrix'`. If `python -c "from dtaidistance import dtw; print(dtw.try_import_c())"` prints `False` or TiVy raises either error, rebuild it against the installed numpy (needs a C compiler):
    ```
    pip install --force-reinstall --no-deps --no-build-isolation --no-binary dtaidistance --no-cache-dir dtaidistance==2.3.13
    ```
    
 4. To use the library, simply copy the `TSVis` folder and `TiVy.py` to your working directory.
 
 ## Usage
 - Basic Usage: Time Series Summarization
 ```python
import numpy as np
from TSVis.synthetic_data import generateSyntheticData
from TiVy import TiVy
from TSVis.widget import TimeSeriesVis

# Generate or load your time series data
time_series = generateSyntheticData(100, 200)  # 100 time series, 200 time points

# Step 1: Initialize TiVy with parameters
tivy = TiVy(
    chunk_size=50,      # window size
    strength=1,         # clustering strength
    use_LSH=True,       # use Locality Sensitive Hashing for speedup
    nPerm=20,          # number of permutations for LSH
    bandWidth=0.5       # bandwidth for LSH
)

# Step 2: Cluster subsequences
tivy.cluster_subsequences(time_series)

# Step 3: Scan for patterns
scan_profile = tivy.prefix_scan(tivy.clusters, tivy.ts_start, minlen=1)

# Step 4: Optimize patterns
counter = scan_profile['counter']
sequenceClusterDB = scan_profile['sequenceClusterDB']
clusters = tivy.optimize_patterns(
    sequenceClusterDB, 
    counter,
    minSupport=10,   # minimum support searched in the prefix scan profile
    minLength=50,    # minimum pattern length
    minSize=1        # minimum number of series to be considered as a pattern
)

# Step 5: Generate visualization output
result = tivy.output_result(clusters)

# Step 6: Visualize the results
summary = TimeSeriesVis(
    n_cols=7,           # number of columns in the grid
    padding=50,         # padding between charts
    n_ticks=3,          # number of tick marks
    chart_width=260,    # chart width in pixels
    chart_height=140,   # chart height in pixels
    interpolate=500     # density map resolution
)
summary.fit(result['result'])
summary.visualize()
 ```
 
 - Direct Visualization (without pattern mining)
```python
from TSVis.widget import TimeSeriesVis
import numpy as np

# Prepare your data as a list of 2D arrays
data = [
    # Four time series in the first chart
    np.array([[1,2,3,4,5],
              [1,0,1,0,1],
              [2,2,2,2,2],
              [5,1,5,1,5]]),
    # Three time series starting from t = 3 in the second chart
    np.array([[np.nan,np.nan,3,4,5],
              [np.nan,np.nan,5,4,3],
              [np.nan,np.nan,2,2,1]]), 
]

# Initialize and visualize
vis = TimeSeriesVis(
    n_cols=3,
    chart_width=300,
    chart_height=150
)
vis.fit(data)
vis.visualize()
```

## Citation
If you use TiVy in your research, please cite our paper:

**Gromit Yeuk-Yin Chan, Luis Gustavo Nonato, Themis Palpanas, Cláudio T. Silva, and Juliana Freire.** "TiVy: Time Series Visual Summary for Scalable Visualization." *IEEE Transactions on Visualization and Computer Graphics (VIS 2025)*.

BibTeX:
```bibtex
@article{chan2025tivy,
  title={TiVy: Time Series Visual Summary for Scalable Visualization},
  author={Chan, Gromit Yeuk-Yin and Nonato, Luis Gustavo and Palpanas, Themis and Silva, Cl{\'a}udio T. and Freire, Juliana},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2026},
  publisher={IEEE}
}
```

## Contact
For more questions or feature requests, feel free to reach me (Gromit Chan) through my [e-mail](mailto:gromit.chan@nyu.edu?subject=[TSVis]%20Source%20Han%20Sans).
