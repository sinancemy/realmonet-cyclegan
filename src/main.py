from . import model as model_
from . import dataset as dataset_

REBUILD_DATASET = False

if REBUILD_DATASET:
    dataset_.build()

data = dataset_.load()
model = model_.Net()
