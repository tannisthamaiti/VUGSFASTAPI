
import multiprocessing

def setup():
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set
        pass

setup()

from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse,HTMLResponse
from pydantic import BaseModel
import os
from vug_extractor import extract_and_plot_contours, plotfmi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from helper import load_depth, load_and_scale, array_to_png, array_to_png_batches_parallel
from utils.data import get_data
import io
import numpy as np
import pandas as pd
import time
from fastapi import Header, Request, Response
import json
import hashlib
from datetime import datetime


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # tighten in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VugRequest(BaseModel):
    
    min_vug_area: float = Field(..., ge=0, description="Minimum vug area")
    max_vug_area: float = Field(..., ge=0, description="Maximum vug area")
    min_circ_ratio: float = Field(..., ge=0, description="Minimum circularity ratio")
    max_circ_ratio: float = Field(..., ge=0, description="Maximum circularity ratio")

from fastapi import HTTPException, Query
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import io
from pathlib import Path

@app.post("/extract-vugs/")
def extractvugs(
    vug_request: VugRequest,
    sha_short: str = Query(..., description="Short SHA ID (first 8 chars)"),
    timestamp: str = Query(..., description="UTC timestamp in format YYYYMMDDTHHMMSSZ")
):
    """
    Extract vugs from pre-processed FMI data identified by SHA + timestamp.
    """

    # Extract parameters from request
    min_area = vug_request.min_vug_area
    max_area = vug_request.max_vug_area
    min_circ = vug_request.min_circ_ratio
    max_circ = vug_request.max_circ_ratio

    # Build file paths
    data_path = Path("/app/well_files")
    fmi_csv = data_path / f"fmi_array_{sha_short}_{timestamp}.csv"
    unscaled_csv = data_path / f"fmi_array_unscaled_{sha_short}_{timestamp}.csv"
    tdep_csv = data_path / f"tdep_array_{sha_short}_{timestamp}.csv"
    radius_csv = data_path / f"well_radius_{sha_short}_{timestamp}.csv"

    try:
        # Load required data
        filtered_data = pd.read_csv(fmi_csv, header=None).astype(float).values
        #filtered_unscaled_data = pd.read_csv(unscaled_csv, header=None).astype(float).values
        filtered_depths = pd.read_csv(tdep_csv, header=None).astype(float).values.squeeze()
        well_radius = pd.read_csv(radius_csv, header=None).astype(float).values.squeeze()

        print(f"[INFO] Loaded shapes — FMI: {filtered_data.shape}, Depths: {filtered_depths.shape}")

        # Optional slicing for performance or testing
        # filtered_data = filtered_data[25000:150000]
        # filtered_depths = filtered_depths[25000:150000]
        # well_radius = well_radius[25000:150000]

        start_depth = filtered_depths.min()
        end_depth = filtered_depths.max()

        # Extract and plot vugs
        png_list = extract_and_plot_contours(
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

        if not png_list:
            raise ValueError("No vug plots were generated.")

        return StreamingResponse(io.BytesIO(png_list[0]), media_type="image/png")

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Required file not found: {e}")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


   

@app.get("/plotfmi")
def plotfmi_handler(
    request: Request,
    response: Response,
    x_metadata: str = Header(...)
):
    """
    PNG heat-map of the full scaled FMI array with SHA256-based tracking and UTC timestamp.
    Saves derived CSVs and logs metadata using SHA+timestamp identifier.
    """

    # === Step 1: Parse metadata from header ===
    try:
        metadata_dict = json.loads(x_metadata)
        print(f"[INFO] Received metadata: {metadata_dict}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in X-Metadata")

    # === Step 2: Compute SHA256 and timestamp ===
    sha_full = hashlib.sha256(x_metadata.encode()).hexdigest()
    sha_short = sha_full[:8]
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")  # Safe, short ISO-like
    response.headers["X-Metadata-SHA256"] = sha_full
    response.headers["X-Server-Timestamp"] = timestamp
    print(f"[INFO] SHA256: {sha_full}, Short: {sha_short}, Timestamp: {timestamp}")

    # === Step 3: Load data ===
    start_time = time.time()
    data_path = Path("/app/well_files")
    metadata_log_path = data_path / "metadata_log.csv"

    try:
        fmi_array, tdep_array, well_radius = get_data(data_path, dyn=True)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data missing: {e}")

    # === Step 4: Preprocess and plot ===
    try:
        DEPTH_VEC = tdep_array
        SCALED_DATA = fmi_array

        fmi_array_doi, _, tdep_array_doi, well_radius_doi = plotfmi(
            data_path=SCALED_DATA,
            depth_path=DEPTH_VEC,
            well_radius=well_radius,
            start_depth=DEPTH_VEC.min(),
            end_depth=DEPTH_VEC.max()
        )

        # Skip NaN rows from top
        start_index = np.argmax(~np.all(np.isnan(fmi_array), axis=1))
        fmi_array_doi = fmi_array_doi[start_index:]
        tdep_array_doi = tdep_array_doi[start_index:]
        well_radius_doi = well_radius_doi[start_index:]

        # === Step 5: Save CSVs with short hash + timestamp in name ===
        fmi_csv = data_path / f"fmi_array_{sha_short}_{timestamp}.csv"
        #unscaled_csv = data_path / f"fmi_array_unscaled_{sha_short}_{timestamp}.csv"
        tdep_csv = data_path / f"tdep_array_{sha_short}_{timestamp}.csv"
        radius_csv = data_path / f"well_radius_{sha_short}_{timestamp}.csv"

        pd.DataFrame(fmi_array_doi).to_csv(fmi_csv, header=False, index=False)
        #pd.DataFrame(fmi_array_doi_unscaled).to_csv(unscaled_csv, header=False, index=False)
        pd.DataFrame(tdep_array_doi).to_csv(tdep_csv, header=False, index=False)
        pd.DataFrame(well_radius_doi).to_csv(radius_csv, header=False, index=False)

        # === Step 6: Log metadata ===
        log_entry = pd.DataFrame([{
            "sha_id": sha_full,
            "sha_short": sha_short,
            "timestamp": timestamp,
            "fmi_array_path": str(fmi_csv),
            "tdep_array_path": str(tdep_csv),
            "well_radius_path": str(radius_csv),
            "original_metadata": x_metadata
        }])
        log_entry.to_csv(metadata_log_path, mode='a', index=False, header=not metadata_log_path.exists())

        # === Step 7: Generate PNG ===
        png_bytes = array_to_png_batches_parallel(fmi_array_doi, tdep_array_doi, batch_size=500)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    duration = time.time() - start_time
    print(f"[TIMER] /plotfmi completed in {duration:.2f} seconds.")

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={
            "X-Metadata-SHA256": sha_full,
            "X-Server-Timestamp": timestamp
        }
    )

    




    


