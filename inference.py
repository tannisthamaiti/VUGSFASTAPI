#@title 4. Download Production Artifacts from W&B (Final Corrected Version)
import wandb
import os

print("--> Connecting to W&B to download production artifacts...")
# Initialize the API to access the project
api = wandb.Api()
entity_project_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"

try:
    # Download the main trained model
    model_artifact_name = f'boundary-detector-model:{MODEL_VERSION}'
    print(f"--> Downloading model: {model_artifact_name}")
    model_artifact = api.artifact(f'{entity_project_path}/{model_artifact_name}', type='model')
    model_artifact.download(root=os.path.dirname(config['paths']['final_model_path']))
    print("✅ Model downloaded successfully.")

    # Download the StandardScaler
    scaler_artifact_name = 'StandardScaler:latest'
    print(f"--> Downloading StandardScaler: {scaler_artifact_name}")
    scaler_artifact = api.artifact(f'{entity_project_path}/{scaler_artifact_name}', type='preprocessor')
    scaler_artifact.download(root=os.path.dirname(config['paths']['std_scaler_path']))
    print("✅ StandardScaler downloaded successfully.")

    # Download the Label Encoder
    encoder_artifact_name = 'LabelEncoder:latest'
    print(f"--> Downloading Label Encoder: {encoder_artifact_name}")
    encoder_artifact = api.artifact(f'{entity_project_path}/{encoder_artifact_name}', type='preprocessor')
    encoder_artifact.download(root=os.path.dirname(config['paths']['label_encoder_path']))
    print("✅ Label Encoder downloaded successfully.")

except Exception as e:
    print("\n" + "="*50)
    print("🚨 ERROR DOWNLOADING ARTIFACTS 🚨")
    print(f"An error occurred: {e}")
    print("\nPLEASE CHECK THE FOLLOWING:")
    print("1. Did you successfully re-run the TRAINING notebook with the corrected Cell 13?")
    print(f"2. Do you see 'StandardScaler' and 'LabelEncoder' in the 'Artifacts' tab of your W&B project: https://wandb.ai/{entity_project_path}/artifacts")
    print(f"3. Are the WANDB_ENTITY and WANDB_PROJECT names in Cell 3 of THIS notebook spelled correctly?")
    print("="*50)

print("\n--- All necessary artifacts should now be available locally. ---")

#@title 7. Define Plotting Logic
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_well_correlation(well1,well2,layers1,layers2,matrix,threshold,path):
    plt.style.use('ggplot'); fig,ax=plt.subplots(figsize=(10,12))
    if not layers1 or not layers2: print(f'Warning: Plotting skipped. Well 1 layers:{len(layers1)}, Well 2:{len(layers2)}'); return
    max_depth=max(layers1[-1]['bottom'],layers2[-1]['bottom']) if layers1 and layers2 else 1000
    ax.set_ylim(max_depth+50,-50); ax.set_xlim(-0.5,2.5)
    n1=len(set(l['Group'] for l in layers1)); n2=len(set(l['Group'] for l in layers2))
    for l in layers1: ax.add_patch(patches.Rectangle((0,l['Top']),1,l['Height'],ec='k',fc=plt.cm.viridis(l['Group']/(n1 if n1>0 else 1)),alpha=0.6))
    for l in layers2: ax.add_patch(patches.Rectangle((1.5,l['Top']),1,l['Height'],ec='k',fc=plt.cm.viridis(l['Group']/(n2 if n2>0 else 1)),alpha=0.6))
    for i,row in enumerate(matrix):
        for j,sim in enumerate(row):
            if sim>=threshold: ax.add_patch(patches.Polygon([[1,layers1[i]['Top']],[1,layers1[i]['bottom']],[1.5,layers2[j]['bottom']],[1.5,layers2[j]['Top']]],fc=plt.cm.Greens(sim),alpha=0.5))
    ax.set_xticks([0.5,2]); ax.set_xticklabels([well1,well2],fontsize=14); ax.set_ylabel('Depth',fontsize=12); ax.set_title('Well to Well Correlation',fontsize=16); plt.savefig(path); plt.close()
    print(f'--> Correlation plot saved to {path}')

def generate_single_correlation_plot(config,full_data,ref_name,woi_name,out_path):
    inf,p=config['inference'],config['paths']; ref_df,woi_df=full_data[full_data['WELL']==ref_name],full_data[full_data['WELL']==woi_name]
    if ref_df.empty or woi_df.empty: print(f"Error: Could not find '{ref_name}' or '{woi_name}'. Please check names."); return False
    with open(p['label_encoder_path']) as f: le=json.load(f)
    def get_layers(df):
        df=df.copy().reset_index(drop=True); df['gid']=df['GROUP'].astype(str).map(le).fillna(-1).astype(int); b=np.where(df['gid'].iloc[:-1].values!=df['gid'].iloc[1:].values)[0]+1
        indices=np.concatenate(([0],b,[len(df)])); layers=[]
        for i in range(len(indices)-1):
            s,e=indices[i],indices[i+1]
            if s<e: layers.append({'Top':df['DEPTH_MD'].iloc[s],'bottom':df['DEPTH_MD'].iloc[e-1],'Height':df['DEPTH_MD'].iloc[e-1]-df['DEPTH_MD'].iloc[s],'Group':df['gid'].iloc[s]})
        return layers
    ref_l,woi_l=get_layers(ref_df),get_layers(woi_df); sim=np.zeros((len(ref_l),len(woi_l)))

    # This is the MOCK INFERENCE part. A real implementation would use the model here.
    print('--> MOCK INFERENCE: Using ground truth layers for visualization.')
    for i,l1 in enumerate(ref_l):
        for j,l2 in enumerate(woi_l): sim[i,j]=np.random.uniform(0.8,0.95) if l1['Group']==l2['Group'] and l1['Group']!=-1 else np.random.uniform(0.1,0.4)

    plot_well_correlation(ref_name,woi_name,ref_l,woi_l,sim,inf['correlation_threshold'],out_path); return True

print("✅ Plotting logic defined.")