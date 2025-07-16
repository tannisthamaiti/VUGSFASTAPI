from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse,HTMLResponse
from pydantic import BaseModel
import os
from vug_extractor import extract_and_plot_contours, plotfmi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from helper import load_depth, load_and_scale, array_to_png
from utils.data import get_data, get_mud_info
import io
import numpy as np


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # tighten in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VugRequest(BaseModel):
    
    min_vug_area: float = Field(..., gt=1, description="min vug area default 1")
    max_vug_area: float = Field(..., gt=100, description="max vug area default 100")
    min_circ_ratio: float = Field(..., gt=0.5, description="min circle ratio default 0.5")
    max_circ_ratio: float = Field(..., gt=1, description="max circle ratio default 1")

# ─── configuration ─────────────────────────────────────────────────────
DATA_PATH  = Path("/app/well_files/fmi_array.csv")   # 2-D array (rows = depth steps)
DEPTH_PATH = Path("/app/well_files/tdep_array.csv")  # 1-D depth log (same nrows)





def get_depth_range(
    start_depth: float = Query(..., gt=0, description="Start depth"),
    end_depth: float = Query(..., gt=0, description="End depth"),
) -> VugRequest:
    if start_depth >= end_depth:
        raise HTTPException(status_code=400, detail="start_depth must be less than end_depth")
    return VugRequest(start_depth=start_depth, end_depth=end_depth)


@app.get("/extract-vugs/")
def extractvugs(vug_request: VugRequest):
    # Now you can access all parameters directly
    min_area = vug_request.min_vug_area
    max_area = vug_request.max_vug_area
    min_circ = vug_request.min_circ_ratio
    max_circ = vug_request.max_circ_ratio
    buf = extract_and_plot_contours(
            data_path=filtered_data,
            depth_path=filtered_depths,
            start_depth=depth.start_depth,
            end_depth=depth.end_depth
)  
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(open(buf, "rb"), media_type="image/png")


@app.get("/plotfmi")
def plotfmi_handler():
    """PNG heat-map of the full scaled FMI array."""
    data_path=Path("/app/well_files")
    MISSING_THRESHOLD = -5000
    fmi_array, tdep_array, well_radius, gt = get_data(data_path, dyn=True)
    print(type(fmi_array), type(tdep_array))
    fmi_array[fmi_array <= MISSING_THRESHOLD] = np.nan
    try:
        DEPTH_VEC   = load_depth(tdep_array)       # (N,)
        SCALED_DATA = load_and_scale(fmi_array)    # (N, M)
        print("[INFO] CSV data loaded at startup.")
    except FileNotFoundError as e:
        DEPTH_VEC = SCALED_DATA = None             # stay None until files appear
        print(f"[WARN] {e} – /plot will 404 until both CSVs exist.")
    
    
    try:
        fmi_array, tdep_array = plotfmi(
            data_path=SCALED_DATA,
            depth_path=DEPTH_VEC,
            well_radius=well_radius,
            start_depth=DEPTH_VEC.min(),
            end_depth=DEPTH_VEC.max()
        )
        png_bytes = array_to_png(fmi_array[:10000], tdep_array[:10000])
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
    




    


