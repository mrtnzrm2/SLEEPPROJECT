import numpy as np
import h5py
import dask.array as da
import logging


from pathlib import Path, PosixPath
from tqdm import tqdm

parent_dir = Path("/data3/mjorge/SLEEPPROJECT/data/20231026/pd54/h5")

logger = logging.getLogger('general')
logger.setLevel(logging.DEBUG)

def realsorted(L : list[PosixPath]) -> list:
  Ldict = dict()
  for l in L:
    ltif0 = l.parts[-1]
    # print(ltif0)
    if "template" in ltif0: continue
    ltif = ltif0.split("Pos")[-1].split(".ome")[0].split("_")
    if len(ltif) == 1:
      Ldict[0] = l
    else:
      Ldict[int(ltif[1])] = l
  Lkeys = list(Ldict.keys())
  Lkeys.sort()
  LdictSorted = {key: Ldict[key] for key in Lkeys}
  return list(LdictSorted.values())
    

def get_number_of_tifs():
    parent_folder = (
        Path("/data3")
        / "mjorge"
        / "SLEEPPROJECT"
        / "data"
        / "20231026"
        / "pd54"
        / "tifRIS"
    )
    filenames = sorted([x for x in parent_folder.glob("*.tif")])
    filenames = realsorted(filenames)
    return len(filenames)

def movie_shape(counter : int):
    parent_folder = (
        Path("/data3")
        / "mjorge"
        / "SLEEPPROJECT"
        / "data"
        / "20231026"
        / "pd54"
        / "h5"
        / "files"
    )

    frames = 0
    for i, f in tqdm(enumerate(parent_folder.glob("*.h5"))):
       if i >= counter: break
       frames += h5py.File(f)["data"].shape[0]

    shape = h5py.File(f)["data"].shape[1:]

    return (frames, ) + shape

if __name__ == "__main__":

  MOVIE_SHAPE = movie_shape(3)
  ndatasets = get_number_of_tifs() # number of tif files

  slc = list(np.arange(0, ndatasets- 2, 2)) + [ndatasets]
  slc_t = [[slc[i], slc[i+1]] for i in np.arange(len(slc)-1)]
  slc_t = [(slc_t[0][0], slc_t[0][1])] + [(i-1, j) for i,j in slc_t[1:]]

  master_darray = da.empty(shape=MOVIE_SHAPE, chunks=(MOVIE_SHAPE[0], 1, 1, 1))

  last_step, current_step = 0, 0

  logger.debug("Start adding data")
  for i in tqdm(np.arange(3)):
    f = h5py.File(parent_dir / "files" / f"data_{slc_t[i][0]}_{slc_t[i][1]-1}.h5")["data"]
    current_step += f.shape[0]
    master_darray[last_step:current_step] = f
    last_step = current_step

  logger.debug("Computing median")
  MEDIAN = da.median(master_darray, axis=0).compute()

  print(MEDIAN)
  print(MEDIAN.shape)