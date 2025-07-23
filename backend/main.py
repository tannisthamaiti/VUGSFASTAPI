
import multiprocessing

def setup():
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set
        pass

setup()

from fastapi import FastAPI, Query, HTTPException, File,Body,UploadFile
from fastapi.responses import FileResponse, StreamingResponse,HTMLResponse
from pydantic import BaseModel
import os
from vug_extractor import extract_and_plot_contours, plotfmi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from helper import array_to_png_batches_parallel,save_contours_to_csv
from utils.data import get_data
import io
import numpy as np
import pandas as pd
import time
from fastapi import Header, Request, Response
import json
import hashlib
from datetime import datetime
import tempfile
import base64


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # tighten in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Metadata-SHA256"]
)


class VugRequest(BaseModel):
    
    min_vug_area: float = 0.5
    max_vug_area: float = 10.0
    min_circ_ratio: float = 0.5
    max_circ_ratio: float = 2.0

from fastapi import HTTPException, Query
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import io
from pathlib import Path

@app.get("/download_vug_csv/{sha_id}")
def download_csv(sha_id: str):
    data_path = Path("/app/well_files")
    csv_file = data_path / f"vug_contours_{sha_id}.csv"

    # Check if the CSV file exists
    if not csv_file.exists():
        raise HTTPException(status_code=404, detail="CSV file not found")

    # Return file as a downloadable response
    return FileResponse(
        path=csv_file,
        media_type='text/csv',
        filename=f"vug_contours_{sha_id}.csv"
    )

    

@app.post("/extract_vugs/")
def extractvugs(
    vug_request: VugRequest = Body(default=VugRequest()),
    sha_short: str = Query(..., description="Short SHA ID (first 8 chars)")
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
    fmi_csv = data_path / f"fmi_array_{sha_short}.csv"
    tdep_csv = data_path / f"tdep_array_{sha_short}.csv"
    radius_csv = data_path / f"well_radius_{sha_short}.csv"

    try:
        # Load required data
        filtered_data = pd.read_csv(fmi_csv, header=None).astype(float).values
        filtered_depths = pd.read_csv(tdep_csv, header=None).astype(float).values.squeeze()
        well_radius = pd.read_csv(radius_csv, header=None).astype(float).values.squeeze()

        print(f"[INFO] Loaded shapes — FMI: {filtered_data.shape}, Depths: {filtered_depths.shape}")

        # Optional slicing for performance or testing
        # filtered_data = filtered_data[25000:26000]
        # filtered_depths = filtered_depths[25000:26000]
        # well_radius = well_radius[25000:26000]

        start_depth = filtered_depths.min()
        end_depth = filtered_depths.max()

        # Extract and plot vugs
        png_list,contour_csv = extract_and_plot_contours(
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
        
        csv_path=save_contours_to_csv(contour_csv, sha_short)
        png_base64 = base64.b64encode(png_list[0]).decode('utf-8')

        return StreamingResponse(io.BytesIO(png_list[0]), media_type="image/png")
       
        

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Required file not found: {e}")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")



@app.post("/process_dlis", tags=["Processing"])
async def process_dlis_handler(
    file: UploadFile = File(..., description="Binary DLIS file to process.")
):
    """
    Accepts a binary DLIS file, saves it temporarily, processes it using a 
    path-based function, and returns a PNG heat-map.
    """
    start_time = time.time()
    # 1. Generate cryptographically strong random data.
    random_data = os.urandom(64)

    # 2. Create a SHA256 hash of the random data.
    full_sha256 = hashlib.sha256(random_data).hexdigest()

    # 3. Take the first 8 characters of the resulting hex string.
    short_sha = full_sha256[:8]


    
    # Read the file content from the upload
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    
    data_path = Path("/app/well_files")

    # Use a temporary directory that is automatically cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create a Path object for the temporary directory
            temp_dir_path = Path(temp_dir)
            
            # Define the full path for the temporary file
            temp_file_path = temp_dir_path / file.filename
            
            # Write the uploaded content to the temporary file
            with open(temp_file_path, 'wb') as f:
                f.write(contents)
            
            print(f"[INFO] File '{file.filename}' temporarily saved to '{temp_dir_path}'")
            
            # === Call your path-based function with the temporary directory path ===
            fmi_array, tdep_array, well_radius = get_data(str(temp_dir_path))
        except FileNotFoundError as e:
            raise HTTPException(status_code=422, detail=f"DLIS processing error: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

           
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
            fmi_csv = data_path / f"fmi_array_{short_sha}.csv"
            tdep_csv = data_path / f"tdep_array_{short_sha}.csv"
            radius_csv = data_path / f"well_radius_{short_sha}.csv"

            pd.DataFrame(fmi_array_doi).to_csv(fmi_csv, header=False, index=False)
            pd.DataFrame(tdep_array_doi).to_csv(tdep_csv, header=False, index=False)
            pd.DataFrame(well_radius_doi).to_csv(radius_csv, header=False, index=False)

            # === Step 6: Log metadata ===
            # log_entry = pd.DataFrame([{
            #     "sha_id": sha_full,
            #     "sha_short": sha_short,
            #     "timestamp": timestamp,
            #     "fmi_array_path": str(fmi_csv),
            #     "tdep_array_path": str(tdep_csv),
            #     "well_radius_path": str(radius_csv),
            #     "original_metadata": x_metadata
            # }])
            # log_entry.to_csv(metadata_log_path, mode='a', index=False, header=not metadata_log_path.exists())

            # === Step 7: Generate PNG ===
            png_bytes = array_to_png_batches_parallel(fmi_array_doi, tdep_array_doi, batch_size=500)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
        
            
    # The temporary directory and its contents are now automatically deleted.

    duration = time.time() - start_time
    print(f"[TIMER] Full request for '{file.filename}' completed in {duration:.2f} seconds.")

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={
            "X-Metadata-SHA256": short_sha
            }
    )

    




    


