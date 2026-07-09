import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Preprocessing ---
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    h, w = img.shape[:2]
    target_aspect = 4/3
    if w/h > target_aspect:
        new_w = int(h*target_aspect)
        crop_x = (w-new_w)//2
        img = img[:, crop_x:crop_x+new_w]
    else:
        new_h = int(w/target_aspect)
        crop_y = (h-new_h)//2
        img = img[crop_y:crop_y+new_h,:]
    h,w = img.shape[:2]
    if w>800 or h>600:
        img = cv2.resize(img, (800,600), interpolation=cv2.INTER_AREA)
        return img
    top = (600-h)//2; bottom = 600-h-top
    left = (800-w)//2; right = 800-w-left
    img = cv2.copyMakeBorder(img, top,bottom,left,right, cv2.BORDER_CONSTANT,value=(0,0,0))
    return img

# --- Compute block features ---
def compute_block_features(block):
    features = {}
    # Saliency
    saliency_obj = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, sal_map = saliency_obj.computeSaliency(block)
    sal_map = (sal_map*255).astype(np.float64)
    features['sal_avg'] = sal_map.mean()
    features['sal_std'] = sal_map.std()
    # HSV
    hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV).astype(np.float64)
    h,s,v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    features['h_mean'], features['s_mean'], features['v_mean'] = h.mean(), s.mean(), v.mean()
    features['h_std'], features['s_std'], features['v_std'] = h.std(), s.std(), v.std()
    # Sobel
    gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY).astype(np.float64)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F,1,0,ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F,0,1,ksize=3)
    features['sobel_mean_x'], features['sobel_std_x'] = sobel_x.mean(), sobel_x.std()
    features['sobel_mean_y'], features['sobel_std_y'] = sobel_y.mean(), sobel_y.std()
    sob_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    features['sobel_mean_mag'], features['sobel_std_mag'] = sob_mag.mean(), sob_mag.std()
    # HSV ratios / contrast
    features['hs_ratio'] = features['h_mean']/(features['s_mean']+1e-6)
    features['hv_contrast'] = np.std(h/(v+1e-6))
    # Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    features['lap_mean'], features['lap_std'] = lap.mean(), lap.std()
    return features

# --- Neighbor features ---
def add_neighbor_features(df, neighbor_cols, n_cols=8):
    df = df.copy()
    lookup = df.set_index(['row','col'])
    for idx,row in df.iterrows():
        r,c = row['row'], row['col']
        neighbors = [(r+dr,c+dc) for dr in [-1,0,1] for dc in [-1,0,1] if not (dr==0 and dc==0) and 0<=r+dr<n_cols and 0<=c+dc<n_cols]
        nbr_vals = []
        for nr,nc in neighbors:
            if (nr,nc) in lookup.index:
                nbr_vals.append(lookup.loc[(nr,nc), neighbor_cols])
        if nbr_vals:
            nbr_df = pd.DataFrame(nbr_vals)
            for col in neighbor_cols:
                df.loc[idx,f"{col}_nbrmean"] = nbr_df[col].mean()
                df.loc[idx,f"{col}_nbrstd"] = nbr_df[col].std()
        else:
            for col in neighbor_cols:
                df.loc[idx,f"{col}_nbrmean"] = np.nan
                df.loc[idx,f"{col}_nbrstd"] = np.nan
    return df

# --- Extract features from a single image ---
def extract_features_from_image(img, img_file):
    blocks=[]
    h,w = img.shape[:2]; bh,bw = h//8, w//8; block_id=0
    for i in range(8):
        for j in range(8):
            block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            feat = compute_block_features(block)
            feat.update({'block_id':block_id,'row':i,'col':j})
            blocks.append(feat)
            block_id+=1
    df = pd.DataFrame(blocks)
    df['filename']=img_file
    return df

# --- Folder processing function ---
def process_image_folder(folder_path, best_model, output_csv="predictions.csv", output_img_folder="predictions"):
    os.makedirs(output_img_folder, exist_ok=True)
    prediction_rows=[]
    
    # Base features for prediction
    base_features = ['row', 'col', 'block_id',

    # base features
    'sal_avg', 'sal_std',
    'h_mean', 's_mean', 'v_mean',
    'h_std', 's_std', 'v_std',
     # neighbor features
    'sal_avg_nbrmean', 'sal_avg_nbrstd',
    'sal_std_nbrmean', 'sal_std_nbrstd',
    'h_mean_nbrmean', 'h_mean_nbrstd',
    's_mean_nbrmean', 's_mean_nbrstd',
    'v_mean_nbrmean', 'v_mean_nbrstd',
    'h_std_nbrmean', 'h_std_nbrstd',
    's_std_nbrmean', 's_std_nbrstd',
    'v_std_nbrmean', 'v_std_nbrstd',
    'sobel_mean_mag', 'sobel_std_mag',
    'sobel_mean_x', 'sobel_std_x',
    'sobel_mean_y', 'sobel_std_y', 
    'sobel_mean_mag_nbrmean', 'sobel_mean_mag_nbrstd',
    'sobel_std_mag_nbrmean',  'sobel_std_mag_nbrstd',
    'sobel_mean_x_nbrmean',   'sobel_mean_x_nbrstd',
    'sobel_std_x_nbrmean',    'sobel_std_x_nbrstd',
    'sobel_mean_y_nbrmean',   'sobel_mean_y_nbrstd',
    'sobel_std_y_nbrmean',    'sobel_std_y_nbrstd', 
       'hs_ratio'	,'hv_contrast']
    
    neighbor_cols = ['sal_avg','sal_std','h_mean','s_mean','v_mean','h_std','s_std', 'v_std',
                     'sobel_mean_x','sobel_std_x','sobel_mean_y','sobel_std_y','sobel_mean_mag','sobel_std_mag',
                     'hs_ratio','hv_contrast','lap_mean','lap_std']

    for img_file in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable image: {img_file}")
            continue
        img = preprocess_image(img_path)
        df_blocks = extract_features_from_image(img, img_file)
        df_blocks = add_neighbor_features(df_blocks, neighbor_cols)
        
        # Predict
        X_blocks = df_blocks[base_features]
        y_pred_blocks = best_model.predict(X_blocks)
        
        # Save predictions row-wise
        pred_row = [img_file] + list(y_pred_blocks)
        prediction_rows.append(pred_row)
        
       # Annotate image
        overlay = img.copy()
        yellow = (0, 255, 255)
        alpha = 0.5
        h, w = img.shape[:2]
        bh, bw = h // 8, w // 8

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (0, 0, 0)  # black text

        for idx, val in enumerate(y_pred_blocks):
            r = idx // 8
            c = idx % 8
            y1, y2 = r * bh, (r + 1) * bh
            x1, x2 = c * bw, (c + 1) * bw

            # Apply yellow overlay if predicted = 1
            if val == 1:
                overlay[y1:y2, x1:x2] = cv2.addWeighted(
                    overlay[y1:y2, x1:x2], 1 - alpha,
                    np.full((y2 - y1, x2 - x1, 3), yellow, dtype=np.uint8), alpha, 0
                )

            # Put block number in the top-left of each block
            text_pos = (x1 + 5, y1 + 15)  # slight offset
            cv2.putText(overlay, str(idx), text_pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Draw white grid
        for i in range(1, 8):
            cv2.line(overlay, (0, i * bh), (w, i * bh), (255, 255, 255), 1)
            cv2.line(overlay, (i * bw, 0), (i * bw, h), (255, 255, 255), 1)

        # Save image
        base_name = os.path.splitext(img_file)[0]
        save_name = base_name + ".png"
        cv2.imwrite(os.path.join(output_img_folder, save_name), overlay)

            
    # Save CSV
    columns = ['filename'] + [f'block{i}' for i in range(64)]
    pred_df = pd.DataFrame(prediction_rows, columns=columns)
    pred_df.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to {output_csv} and annotated images to {output_img_folder}")
    return pred_df

class BlockClassifier:
    def __init__(self, model):
        self.model = model
    
    def predict_folder(self, folder_path, output_csv="predictions.csv", output_img_folder="predictions"):
        
       return (process_image_folder(folder_path, self.model, output_csv, output_img_folder))
