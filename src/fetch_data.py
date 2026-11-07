import os
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("thesis-gb7nx").project("flying-objects-yulsy")
project.version(7).download("coco", location="data/raw")