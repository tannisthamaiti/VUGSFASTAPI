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

@app.post("/extract-vugs/")
def extractvugs(vug_request: VugRequest):
    # Extract parameters from request
    min_area = vug_request.min_vug_area
    max_area = vug_request.max_vug_area
    min_circ = vug_request.min_circ_ratio
    max_circ = vug_request.max_circ_ratio
    print(f"[DEBUG] Received parameters:")
    print(f"  min_vug_area: {min_area}")
    print(f"  max_vug_area: {max_area}")
    print(f"  min_circ_ratio: {min_circ}")
    print(f"  max_circ_ratio: {max_circ}")

    try:
        # Load input data
        filtered_data = pd.read_csv("/app/well_files/fmi_array.csv", header=None).astype(float).values
        filtered_unscaled_data=pd.read_csv("/app/well_files/fmi_array_unscaled.csv", header=None).astype(float).values
        filtered_depths = pd.read_csv("/app/well_files/tdep_array.csv", header=None).astype(float).values.squeeze()
        well_radius = pd.read_csv("/app/well_files/well_radius.csv", header=None).astype(float).values.squeeze()
        print(f"[INFO] FMI shape: {filtered_data.shape}, Depths: {filtered_depths.shape}")
        start_depth = filtered_depths.min()
        end_depth = filtered_depths.max()

        # Perform contour extraction
        png_list = extract_and_plot_contours(
            data_path=filtered_data,
            data_path_unscaled=filtered_unscaled_data,
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
            raise ValueError("No plots generated")

        # Return the first PNG image as a streaming response
        return StreamingResponse(io.BytesIO(png_list[0]), media_type="image/png")

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    

@app.get("/plotfmi")
def plotfmi_handler(
    request: Request,
    response: Response,
    x_metadata: str = Header(...)
):
    """
    PNG heat-map of the full scaled FMI array with server-side SHA256 verification
    and UTC timestamp in response headers.
    """

    # Parse metadata JSON
    try:
        metadata_dict = json.loads(x_metadata)
        print(f"[INFO] Received metadata: {metadata_dict}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in X-Metadata")

    # Compute SHA256 hash server-side (for internal logging/validation)
    computed_sha = hashlib.sha256(x_metadata.encode()).hexdigest()
    print(f"[INFO] Computed SHA256 of metadata: {computed_sha}")

    # Optionally include the computed SHA in response headers (for clients to check)
    response.headers["X-Metadata-SHA256"] = computed_sha

    # Add server-side UTC timestamp
    server_timestamp = datetime.utcnow().isoformat() + "Z"
    response.headers["X-Server-Timestamp"] = server_timestamp
    start_time = time.time()  
    data_path = Path("/app/well_files")
    MISSING_THRESHOLD = -5000

    # Load raw data
    fmi_array, tdep_array, well_radius, gt = get_data(data_path, dyn=True)




    # Preprocess
    #fmi_array[fmi_array <= MISSING_THRESHOLD] = np.nan
    try:
        DEPTH_VEC = tdep_array       # (N,)
        SCALED_DATA = fmi_array  # (N, M)
        print("[INFO] CSV data loaded at startup.")
    except FileNotFoundError as e:
        DEPTH_VEC = SCALED_DATA = None
        print(f"[WARN] {e} – /plot will 404 until both CSVs exist.")

    try:
        fmi_array_doi,fmi_array_doi_unscaled,tdep_array_doi, well_radius_doi = plotfmi(
            data_path=SCALED_DATA,
            depth_path=DEPTH_VEC,
            well_radius=well_radius,
            start_depth=DEPTH_VEC.min(),
            end_depth=DEPTH_VEC.max()
        )
         # Save CSVs to data_path
        (data_path / "fmi_array.csv").write_text(
            pd.DataFrame(fmi_array_doi).to_csv(header=False, index=False)
        )
        (data_path / "fmi_array_unscaled.csv").write_text(
            pd.DataFrame(fmi_array_doi_unscaled).to_csv(header=False, index=False)
        )
        (data_path / "tdep_array.csv").write_text(
            pd.DataFrame(tdep_array_doi).to_csv(header=False, index=False)
        )
        
        (data_path / "well_radius.csv").write_text(
            pd.DataFrame(well_radius_doi).to_csv(header=False, index=False)
        )
        #png_bytes = array_to_png(fmi_array, tdep_array)
        png_bytes = array_to_png_batches_parallel(fmi_array_doi, tdep_array_doi, batch_size=1500)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    end_time = time.time()  
    duration = end_time - start_time
    print(f"[TIMER] /plotfmi completed in {duration:.2f} seconds.")


    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")

    




    


