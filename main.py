import streamlit as st
from PIL import Image
from pathlib import Path
import numpy as np
import tempfile
import io

st.header('Image Colorization')

uploaded_image = st.file_uploader('Choose an image..',type=['png', 'jpg','jpeg','webp'])

if uploaded_image is not None:
	with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
		fp = Path(tmp_file.name)
		fp.write_bytes(uploaded_image.getvalue())
		#st.write(tmp_file.name)
		#st.write(tmpdir)
		#st.write(type(tmpdir))
		#st.write(Path(uploaded_image.name))
		#st.write(str(Path(uploaded_image.name)))
		#st.write(tmpdir+'/'+ str(Path(uploaded_image.name)))
		#with open(tmpdir+'/'+ str(Path(uploaded_image.name)), 'wb') as fp:
			#fp.write(uploaded_image.content)
		#st.markdown("## Original PDF file")
		#fp = Path(tmp_file.name)
		#fp.write_bytes(uploaded_image.getvalue())
		#st.write(tmp_file.name)

	
	#fp = Path(uploaded_image.name)
 
	#file_details = {"FileName":uploaded_image.name,"FileType":uploaded_image.type}
	#st.write(file_details)
	#CDIR = os.getcwd()
	#upimg = Image.open(uploaded_image)
	#st.image(img,height=250,width=250)
	#with open(os.path.join(CDIR, uploaded_image.name),'wb') as f:
		#f.write(uploaded_image.getbuffer())
	#st.success('saved')
	
	#img_path = uploaded_image.name#.suffix
	#st.write(Path(uploaded_image.name))
	#bytes_data = uploaded_image.read()
	#image = Image.open(io.BytesIO(bytes_data))
	#st.write("filename:", uploaded_image.name)
	#st.image(image)
	#st.write(uploaded_image)
	#st.write(type(uploaded_image))
	#st.write(opt)
	
	#_img = Image.open(uploaded_image)
	#with io.BytesIO() as output:
		#_img.save(output, format='JPEG')
		#binary_img = output.getvalue()
		#st.write(binary_image.name)


	import argparse
	import matplotlib.pyplot as plt

	from colorizers import *
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-i','--img_path', type=str, default = tmp_file.name)
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
	    st.write("Original")
	    st.image(img, use_column_width=True)
	with col2:
	    st.write("ECCV 16")
	    st.image(out_img_eccv16, use_column_width=True)
	with col3:
	    st.write("SIGGRAPH 17")
	    st.image(out_img_siggraph17, use_column_width=True)


st.subheader('EXAMPLES')
st.markdown('each explanation is from ClipCap')

st.write('A man with a beard and glasses standing in front of a table.')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/b1.jpg", use_column_width=True)
with col2:
	st.image("imgs/b2.jpg", use_column_width=True)
with col3:
	st.image("imgs/b3.jpg", use_column_width=True)


st.write('A river with a bridge crossing it and a forest in the background.')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/a1.jpg", use_column_width=True)
with col2:
	st.image("imgs/a2.jpg", use_column_width=True)
with col3:
	st.image("imgs/a3.jpg", use_column_width=True)


st.write('A woman taking a picture of herself with a camera.')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/d1.jpg", use_column_width=True)
with col2:
	st.image("imgs/d2.jpg", use_column_width=True)
with col3:
	st.image("imgs/d3.jpg", use_column_width=True)


st.write('A woman sitting in a car looking out the window.')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/e1.jpg", use_column_width=True)
with col2:
	st.image("imgs/e2.jpg", use_column_width=True)
with col3:
	st.image("imgs/e3.jpg", use_column_width=True)


st.write('A woman sitting on a bed with a quilt.')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/c1.jpg", use_column_width=True)
with col2:
	st.image("imgs/c2.jpg", use_column_width=True)
with col3:
	st.image("imgs/c3.jpg", use_column_width=True)
	
	
st.write('A large clock tower with a circular ring around it.')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/f1.jpg", use_column_width=True)
with col2:
	st.image("imgs/f2.jpg", use_column_width=True)
with col3:
	st.image("imgs/f3.jpg", use_column_width=True)


st.write('A woman wearing a black hat and a black coat.')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/g1.jpg", use_column_width=True)
with col2:
	st.image("imgs/g2.jpg", use_column_width=True)
with col3:
	st.image("imgs/g3.jpg", use_column_width=True)


st.write('A cat sitting on a table next to a potted plant.')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/h1.jpg", use_column_width=True)
with col2:
	st.image("imgs/h2.jpg", use_column_width=True)
with col3:
	st.image("imgs/h3.jpg", use_column_width=True)


st.write('A tall building with lots of signs on it.')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/i1.jpg", use_column_width=True)
with col2:
	st.image("imgs/i2.jpg", use_column_width=True)
with col3:
	st.image("imgs/i3.jpg", use_column_width=True)


st.write('A building with a lot of windows and a ceiling fan.')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/j1.jpg", use_column_width=True)
with col2:
	st.image("imgs/j2.jpg", use_column_width=True)
with col3:
	st.image("imgs/j3.jpg", use_column_width=True)

	
st.write('A city street filled with lots of traffic.')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/k1.jpg", use_column_width=True)
with col2:
	st.image("imgs/k2.jpg", use_column_width=True)
with col3:
	st.image("imgs/k3.jpg", use_column_width=True)


st.write('A train is traveling down the tracks in a city.')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/l1.jpg", use_column_width=True)
with col2:
	st.image("imgs/l2.jpg", use_column_width=True)
with col3:
	st.image("imgs/l3.jpg", use_column_width=True)

	
st.write('A large building with a clock tower in the background.')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/m1.jpg", use_column_width=True)
with col2:
	st.image("imgs/m2.jpg", use_column_width=True)
with col3:
	st.image("imgs/m3.jpg", use_column_width=True)


st.write('A motorcycle parked on a street next to a building.')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/o1.jpg", use_column_width=True)
with col2:
	st.image("imgs/o2.jpg", use_column_width=True)
with col3:
	st.image("imgs/o3.jpg", use_column_width=True)
	
	
st.write('ECCV : The European Conference on Computer Vision is a biennial research conference with the proceedings published by Springer Science+Business Media. Similar to ICCV in scope and quality, it is held those years which ICCV is not. It is considered to be one of the top conferences in computer vision, alongside CVPR and ICCV, with an A rating from the Australian Ranking of ICT Conferences and an A1 rating from the Brazilian ministry of education. The acceptance rate for ECCV 2010 was 24.4% for posters and 3.3% for oral presentations.')
st.write('SIGGRAPH2017 : SIGGRAPH is the world’s largest, most influential annual conference and exhibition in computer graphics and interactive techniques: Five days of research results, demos, educational sessions, art, screenings, and hands-on interactivity featuring the community’s latest technical achievements, and three days of commercial exhibits displaying the industrys current hardware, software, and services.')
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

