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
import pandas as pd
import time


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
    max_vug_area: float = Field(..., et=100, description="max vug area default 100")
    min_circ_ratio: float = Field(..., gt=0.5, description="min circle ratio default 0.5")
    max_circ_ratio: float = Field(..., gt=1, description="max circle ratio default 1")

@app.post("/extract-vugs/")
def extractvugs(vug_request: VugRequest):
    # Extract parameters from the request
    min_area = vug_request.min_vug_area
    max_area = vug_request.max_vug_area
    min_circ = vug_request.min_circ_ratio
    max_circ = vug_request.max_circ_ratio

    try:
        # Load filtered FMI and depth data
        filtered_data = pd.read_csv("/app/well_files/fmi_array.csv", header=None).astype(float).values
        filtered_depths = pd.read_csv("/app/well_files/tdep_array.csv", header=None).astype(float).values.squeeze()
        well_radius = pd.read_csv("/app/well_files/well_radius.csv", header=None).astype(float).values.squeeze()
        # Derive depth range from filtered depths
        start_depth = filtered_depths.min()
        end_depth = filtered_depths.max()

        # Call your contour extraction function
        buf = extract_and_plot_contours(
            data_path=filtered_data,
            depth_path=filtered_depths,
            well_radius=well_radius,
            start_depth=start_depth,
            end_depth=end_depth,
            min_vug_area=min_area,
            max_vug_area=max_area,
            min_circ_ratio=min_circ,
            max_circ_ratio=max_circ
        )

    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return StreamingResponse(open(buf, "rb"), media_type="image/png")

@app.get("/plotfmi")
def plotfmi_handler():
    """PNG heat-map of the full scaled FMI array."""
    start_time = time.time()  
    data_path = Path("/app/well_files")
    MISSING_THRESHOLD = -5000

    # Load raw data
    fmi_array, tdep_array, well_radius, gt = get_data(data_path, dyn=True)

    # Save to CSV files
    # Save CSVs to data_path
    (data_path / "fmi_array.csv").write_text(
        pd.DataFrame(fmi_array).to_csv(header=False, index=False)
    )
    (data_path / "tdep_array.csv").write_text(
        pd.DataFrame(tdep_array).to_csv(header=False, index=False)
    )
    (data_path / "well_radius.csv").write_text(
        pd.DataFrame(well_radius).to_csv(header=False, index=False)
    )


    # Preprocess
    #fmi_array[fmi_array <= MISSING_THRESHOLD] = np.nan
    try:
        DEPTH_VEC = load_depth(tdep_array)       # (N,)
        SCALED_DATA = load_and_scale(fmi_array)  # (N, M)
        print("[INFO] CSV data loaded at startup.")
    except FileNotFoundError as e:
        DEPTH_VEC = SCALED_DATA = None
        print(f"[WARN] {e} – /plot will 404 until both CSVs exist.")

    try:
        fmi_array, tdep_array = plotfmi(
            data_path=SCALED_DATA,
            depth_path=DEPTH_VEC,
            well_radius=well_radius,
            start_depth=DEPTH_VEC.min(),
            end_depth=DEPTH_VEC.max()
        )
        png_bytes = array_to_png(fmi_array, tdep_array)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    end_time = time.time()  
    duration = end_time - start_time
    print(f"[TIMER] /plotfmi completed in {duration:.2f} seconds.")


    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")

    




    


