import streamlit as st
from PIL import Image

st.title('Image Colorization')

image = Image.open('imgs/b.jpeg')


import argparse
import matplotlib.pyplot as plt

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/b.jpeg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_eccv16.cuda()
	colorizer_siggraph17.cuda()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
if(opt.use_gpu):
	tens_l_rs = tens_l_rs.cuda()

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())


col1, col2, col3 = st.columns(3)

with col1:
    st.header("Original")
    st.image("imgs/ansel_adams3.jpg", use_column_width=True)

with col2:
    st.header("ECCV 16")
    st.image(out_img_eccv16, use_column_width=True)
    
with col3:
    st.header("SIGGRAPH 17")
    st.image(out_img_siggraph17, use_column_width=True)

#st.image(image, caption='Original',use_column_width=True)
#st.image(out_img_eccv16, caption='ECCV 16',use_column_width=True)
#st.image(out_img_siggraph17, caption='SIGGRAPH 17',use_column_width=True)


#plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)
#plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)

#plt.figure(figsize=(12,8))
#plt.subplot(2,2,1)
#plt.imshow(img)
#plt.title('Original')
#plt.axis('off')

#plt.subplot(2,2,2)
#plt.imshow(img_bw)
#plt.title('Input')
#plt.axis('off')

#plt.subplot(2,2,3)
#plt.imshow(out_img_eccv16)
#plt.title('Output (ECCV 16)')
#plt.axis('off')

#plt.subplot(2,2,4)
#plt.imshow(out_img_siggraph17)
#plt.title('Output (SIGGRAPH 17)')
#plt.axis('off')
#plt.show()

