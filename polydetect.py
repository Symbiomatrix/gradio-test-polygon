# GPT3 code. Nonsensical on the sketch.

import cv2
import numpy as np
import gradio as gr
import colorsys
import os
# from PIL import Image

POLYFACTOR = 1.5 # Small lines are detected as shapes.
CCOLOUR = 0 # Should be a form controlled thing.
COLREG = None # Computed colour regions cache. Array. Extended whenever a new colour is requested. 
COLUSE = set() # Used colours. Reset on new canvas / upload.
IDIM = 256
CBLACK = 255
MAXCOL = 360 - 1 # Hsv goes by degrees.
VARIANT = 0 # Ensures that the sketch canvas is actually refreshed.
HSV_RANGE = (125,130) # Permitted hsv error range. Mind, wrong hue might throw off the mask entirely.
HSV_VAL = 128
FEXT = ".png"

# BREAKTHROUGH:
# Sketch can be overridden via controlnet method of creation, an np array with type,
# when varying the shape a bit.

def get_colours(img):
    """List colours used in image (as nxc array).
    
    """
    return np.unique(img.reshape(-1, img.shape[-1]), axis=0)

def generate_unique_colors(n):
    """Generate n visually distinct colors as a list of RGB tuples.
    
    Uses the hue of hsv, with balanced saturation & value.
    """
    hsv_colors = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    rgb_colors = [tuple(int(i * CBLACK) for i in colorsys.hsv_to_rgb(*hsv)) for hsv in hsv_colors]
    return rgb_colors

def deterministic_colours(n, lcol = None):
    """Generate n visually distinct & consistent colours as a list of RGB tuples.
    
    Uses the hue of hsv, with balanced saturation & value.
    Goes around the cyclical 0-256 and picks each /2 value for every round.
    Continuation rules: If pcyv != ccyv in next round, then we don't care.
    If pcyv == ccyv, we want to get the cval + delta of last elem.
    If lcol > n, will return it as is.
    """
    if n <= 0:
        return None
    pcyc = -1
    cval = 0
    if lcol is None:
        st = 0
    elif n <= len(lcol):
        # return lcol[:n] # Truncating the list is accurate, but pointless.
        return lcol
    else:
        st = len(lcol)
        if st > 0:
            pcyc = np.ceil(np.log2(st))
            # This is erroneous on st=2^n, but we don't care.
            dlt = 1 / (2 ** pcyc)
            cval = dlt + 2 * dlt * (st % (2 ** (pcyc - 1)) - 1)

    lhsv = []
    for i in range(st,n):
        ccyc = np.ceil(np.log2(i + 1))
        if ccyc == 0: # First col = 0.
            cval = 0
            pcyc = ccyc
        elif pcyc != ccyc: # New cycle, start from the half point between 0 and first point.
            dlt = 1 / (2 ** ccyc)
            cval = dlt
            pcyc = ccyc
        else:
            cval = cval + 2 * dlt # Jumps over existing vals.
        lhsv.append(cval)
    lhsv = [(v, 0.5, 0.5) for v in lhsv] # Hsv conversion only works 0:1.
    lrgb = [colorsys.hsv_to_rgb(*hsv) for hsv in lhsv]
    lrgb = (np.array(lrgb) * (CBLACK + 1)).astype(np.uint8) # Convert to colour uints.
    lrgb = lrgb.reshape(-1, 3)
    if lcol is not None:
        lrgb = np.concatenate([lcol, lrgb])
    return lrgb

def detect_image_colours(img):
    """Detect relevant hsv colours in image and clean up the mask.
    
    BUG: Rgb->hsb and back is not lossless in np / cv. Getting 128->127.
    Looks like the only option is to use colorsys which is contiguous.
    It's ~x10 slower but the conversion is only on load.
    """
    global COLUSE
    (h,w,c) = img.shape
    # Convert to hsv. CONT: Not actually sure if it's correct. Should test.
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Flatten the image hw and find unique colours.
    lucols = get_colours(hsv_img)
    # Filter colours to the ones we detect only.
    # Allows some leeway with mask deterioration.
    msk = ((lucols[:,1] >= HSV_RANGE[0]) & (lucols[:,1] <= HSV_RANGE[1]) &
           (lucols[:,2] >= HSV_RANGE[0]) & (lucols[:,2] <= HSV_RANGE[1])) 
    lfltcols = lucols[msk]
    # If there are invalid colours (besides bg white), warn that they'll be removed.
    if len(lfltcols) < len(lucols) - 1:
        print("Warning: Invalid colours detected in mask, will be removed.")
    # Find regions which contain the right colours.
    msk2 = np.isin(hsv_img.reshape(-1,c),lfltcols).reshape(h,w,c).all(axis = 2)
    hsv_img[msk2,1:] = HSV_VAL # Make all relevant colours precise values.
    hsv_img[~msk2] = [0,0,CBLACK] # CONT Empty all irrelevant colours.
    # Save the colours used in mask, return img to rgb.
    # First convert to exact vals + hashable tuples, and update in a new set.
    lfltcols[:,1:] = HSV_VAL
    lfltcols2 = [tuple(x) for x in lfltcols] # tolist
    COLUSE = set()
    COLUSE.update(lfltcols2)
    skimg = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    
    return skimg

def save_mask(img, flpt, flnm):
    """Save mask to file.
    
    These will be loaded as part of a preset.
    Cv's colour scheme is an annoyance, but avoiding yet another import. 
    """
    # Cv's colour scheme is annoying.
    try:
        img = img["image"]
    except Exception:
        pass
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)
    cv2.imwrite(os.path.join(os.getcwd(),flpt,flnm + FEXT), img)

def detect_polygons(img,num):
    global CCOLOUR
    global COLREG
    global VARIANT
    global COLUSE
    
    # I dunno why, but mask has a 4th colour channel, which contains nothing. Alpha?
    if VARIANT != 0:
        out = img["image"][:-VARIANT,:-VARIANT,:3]
        img = img["mask"][:-VARIANT,:-VARIANT,:3]
    else:
        out = img["image"][:,:,:3]
        img = img["mask"][:,:,:3]
    
    # Convert the binary image to grayscale
    if img is None:
        img = np.zeros([IDIM,IDIM,3],dtype = np.uint8) + CBLACK # Stupid cv.
    if out is None:
        out = np.zeros_like(img) + CBLACK # Stupid cv.
    bimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Find contours in the image
    # Must reverse colours, otherwise draws an outer box (0->255). Dunno why gradio uses 255 for white anyway. 
    contours, hierarchy = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #img2 = np.zeros_like(img) + 255 # Fresh image.
    img2 = out # Update current image.
    # color = np.random.randint(0,255,3)
    # color = deterministic_colours(CCOLOUR + 1)[-1]
    # CCOLOUR = CCOLOUR +1

    COLREG = deterministic_colours(int(num) + 1, COLREG)
    color = COLREG[int(num),:]
    COLUSE.add(tuple(color))
    # Loop through each contour and detect polygons
    for cnt in contours:
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(cnt, 0.0001 * cv2.arcLength(cnt, True), True)

        # If the polygon has 3 or more sides and is fully enclosed, fill it with a random color
        # if len(approx) >= 3: # BAD test.
        if cv2.contourArea(cnt) > cv2.arcLength(cnt, True) * POLYFACTOR: # Better, still messes up on large brush.
            #SBM BUGGY, prevents contours from . cv2.pointPolygonTest(approx, (approx[0][0][0], approx[0][0][1]), False) >= 0:
            # Check if the polygon has already been filled
            # if i not in filled_polygons: # USELESS
            
            # Draw the polygon on the image with a new random color
            color = [int(v) for v in color] # Opencv is dumb / C based and can't handle an int64 array.
            #cv2.drawContours(img2, [approx], 0, color = color) # Only outer sketch.
            cv2.fillPoly(img2,[approx],color = color)
                

                # Add the polygon to the set of filled polygons
                # filled_polygons.add(i)

    # Convert the grayscale image back to RGB
    #img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB) # Converting to grayscale is dumb.
    
    skimg = create_canvas(img2.shape[0], img2.shape[1])
    if VARIANT != 0:
        skimg[:-VARIANT,:-VARIANT,:] = img2
    else:
        skimg[:,:,:] = img2
    
    return skimg, num + 1 if num + 1 <= CBLACK else num

def detect_mask(img,num):
    color = deterministic_colours(int(num) + 1)[-1]
    color = color.reshape([1,1,3])
    mask = ((img["image"] == color).all(-1)) * CBLACK
    return mask

def create_canvas(h, w):
    """New canvas area.
    
    Small variant value is added (and ignored later) due to gradio refresh bug.
    """
    global VARIANT
    global COLUSE
    COLUSE = set()
    VARIANT = 1 - VARIANT
    vret =  np.zeros(shape=(h + VARIANT, w + VARIANT, 3), dtype=np.uint8) + CBLACK
    return vret

# Define the Gradio interface

# Create the Gradio interface and link it to the polygon detection function
# gr.Interface(detect_polygons, inputs=[sketch,output], outputs=output, title="Polygon Detection",
#               description="Detect and fill closed shapes with different random colors.").launch()

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # Gradio shape is dumb.
            # sketch = gr.Image(shape=(IDIM, IDIM),source = "canvas", tool = "color-sketch")#,brush_radius = 1) # No brush radius in 16.2.
            sketch = gr.Image(source = "upload", mirror_webcam = False, type = "numpy", tool = "sketch")
            # sketch = gr.Image(shape=(256, 256),source = "upload", tool = "color-sketch")
            #num = gr.Number(value = 0)
            num = gr.Slider(label="Region", minimum=0, maximum=MAXCOL, step=1, value=0)
            btn = gr.Button(value = "Draw region")
            btn2 = gr.Button(value = "Display mask")
            canvas_width = gr.Slider(label="Canvas Width", minimum=64, maximum=2048, value=512, step=8)
            canvas_height = gr.Slider(label="Canvas Height", minimum=64, maximum=2048, value=512, step=8)
            cbtn = gr.Button(value="Create mask area")
            # CONT: Awaiting fix for https://github.com/gradio-app/gradio/issues/4088.
            # Upload button kinda sucks. Just gonna make a second image.
            # mskbtn = gr.UploadButton("Upload mask cus gradio", file_types=["image"])
            uploadme = gr.Image(label="Upload mask here cus gradio",source = "upload", type = "numpy")
            dlbtn = gr.Button(value="Download mask")
        with gr.Column():
            # Cannot update sketch in 16.2, must add to different image.
            # output = gr.Image(shape=(IDIM, IDIM), source = "upload")
            # output = gr.Image(source = "upload")
            output2 = gr.Image(shape=(IDIM, IDIM))
    
    btn.click(detect_polygons, inputs = [sketch,num], outputs = [sketch,num])
    btn2.click(detect_mask, inputs = [sketch,num], outputs = [output2])
    cbtn.click(fn=create_canvas, inputs=[canvas_height, canvas_width], outputs=[sketch])
    #sketch.upload(fn = detect_image_colours, inputs = [sketch], outputs = [sketch]) # Unusable until #4088 fixed.
    uploadme.upload(fn = detect_image_colours, inputs = [uploadme], outputs = [sketch])
    dlbtn.click(fn = lambda x: save_mask(x,r"test","tpreset"), inputs = [sketch],outputs = [])

if __name__ == "__main__":
    demo.launch()
