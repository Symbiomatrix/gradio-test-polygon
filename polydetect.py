# GPT3 code. Nonsensical on the sketch.

import cv2
import numpy as np
import gradio as gr
import colorsys

POLYFACTOR = 1.5 # Small lines are detected as shapes.
CCOLOUR = 0 # Should be a form controlled thing.
COLREG = None # Computer colour regions. Array. Extended whenever a new colour is requested. 
IDIM = 256
CBLACK = 255
VARIANT = 0 # Ensures that the sketch canvas is actually refreshed.

# BREAKTHROUGH:
# Sketch can be overridden via controlnet method of creation, an np array with type,
# when varying the shape a bit.

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

def detect_polygons(img,num):
    global CCOLOUR
    global COLREG
    global VARIANT
    
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
            num = gr.Slider(label="Region", minimum=0, maximum=CBLACK, step=1, value=0)
            btn = gr.Button(value = "Draw region")
            btn2 = gr.Button(value = "Display mask")
            canvas_width = gr.Slider(label="Canvas Width", minimum=64, maximum=2048, value=512, step=8)
            canvas_height = gr.Slider(label="Canvas Height", minimum=64, maximum=2048, value=512, step=8)
            cbtn = gr.Button(value="Create mask area")
        with gr.Column():
            # Cannot update sketch in 16.2, must add to different image.
            # output = gr.Image(shape=(IDIM, IDIM), source = "upload")
            # output = gr.Image(source = "upload")
            output2 = gr.Image(shape=(IDIM, IDIM))
    
    btn.click(detect_polygons, inputs = [sketch,num], outputs = [sketch,num])
    btn2.click(detect_mask, inputs = [sketch,num], outputs = [output2])
    cbtn.click(fn=create_canvas, inputs=[canvas_height, canvas_width], outputs=[sketch])

if __name__ == "__main__":
    demo.launch()
