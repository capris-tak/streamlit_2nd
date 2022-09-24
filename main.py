import streamlit as st
from PIL import Image
from pathlib import Path
import numpy as np
import tempfile
import io
from IPython
from colorizers import *

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
	###img = Image.open(file).convert('RGB') new_img = img.copy() new_img.thumbnail((h_size, v_size)) new_img_size = os.path.getsize(file) / 1000
		# 画質を調整 # png if img_format == 'png': for i in range(7,10): new_img.save(conv_folder + file_name + '.png', compress_level=i)
		#new_img_size = os.path.getsize(conv_folder + file_name + '.png') / 1000 if new_img_size > Max_size and i < 9: print(str(new_img_size) + 'KB '+ 'quality=' + str(i))
		#os.remove(conv_folder + file_name + '.png') else: print('画質調整終了 ' + str(new_img_size) + 'KB '+ 'compress_level=' + str(i)) break # jpg elif img_format == 'jpg': for i in range(75, -1, -5): new_img.save(conv_folder + file_name + '.jpg', quality=i)
		#new_img_size = os.path.getsize(conv_folder + file_name + '.jpg') / 1000 if new_img_size > Max_size and i > 0: print(str(new_img_size) + 'KB '+ 'quality=' + str(i))
		#os.remove(conv_folder + file_name + '.jpg') else: print('画質調整終了 ' + str(new_img_size) + 'KB '+ 'quality=' + str(i)) break # webp elif img_format == 'webp': for i in range(80, 0, -5): new_img.save(conv_folder + file_name + '.webp', quality=i)
		#new_img_size = os.path.getsize(conv_folder + file_name + '.webp') / 1000 if new_img_size > Max_size and i > 5: print(str(new_img_size) + 'KB '+ 'quality=' + str(i)) os.remove(conv_folder + file_name + '.webp') else: print('画質調整終了 ' + str(new_img_size) + 'KB '+ 'quality=' + str(i)) break
	#_img = Image.open(uploaded_image)
	#with io.BytesIO() as output:
		#_img.save(output, format='JPEG')
		#binary_img = output.getvalue()
		#st.write(binary_image.name)


	import argparse
	import matplotlib.pyplot as plt

	
	
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
st.write('each explanation from ClipCap(CLIP model + GPT2 tokenizer) and Japanese from googletrans.')

st.caption('A man with a beard and glasses standing in front of a table. テーブルの前に立っているひげと眼鏡を持った男。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/b1.jpg", use_column_width=True)
with col2:
	st.image("imgs/b2.jpg", use_column_width=True)
with col3:
	st.image("imgs/b3.jpg", use_column_width=True)


st.caption('A river with a bridge crossing it and a forest in the background.それを横切る橋がある川と背景に森があります。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/a1.jpg", use_column_width=True)
with col2:
	st.image("imgs/a2.jpg", use_column_width=True)
with col3:
	st.image("imgs/a3.jpg", use_column_width=True)


st.caption('A woman taking a picture of herself with a camera.カメラで自分の写真を撮っている女性。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/d1.jpg", use_column_width=True)
with col2:
	st.image("imgs/d2.jpg", use_column_width=True)
with col3:
	st.image("imgs/d3.jpg", use_column_width=True)


st.caption('A woman sitting in a car looking out the window.窓の外を見ている車に座っている女性。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/e1.jpg", use_column_width=True)
with col2:
	st.image("imgs/e2.jpg", use_column_width=True)
with col3:
	st.image("imgs/e3.jpg", use_column_width=True)


st.caption('A woman sitting on a bed with a quilt.キルトと一緒にベッドに座っている女性。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/c1.jpg", use_column_width=True)
with col2:
	st.image("imgs/c2.jpg", use_column_width=True)
with col3:
	st.image("imgs/c3.jpg", use_column_width=True)
	
	
st.caption('A large clock tower with a circular ring around it.周囲に円形のリングが付いた大きな時計塔。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/f1.jpg", use_column_width=True)
with col2:
	st.image("imgs/f2.jpg", use_column_width=True)
with col3:
	st.image("imgs/f3.jpg", use_column_width=True)


st.caption('A woman wearing a black hat and a black coat.黒い帽子と黒いコートを着た女性。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/g1.jpg", use_column_width=True)
with col2:
	st.image("imgs/g2.jpg", use_column_width=True)
with col3:
	st.image("imgs/g3.jpg", use_column_width=True)


st.caption('A cat sitting on a table next to a potted plant.鉢植えの植物の隣のテーブルの上に座っている猫。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/h1.jpg", use_column_width=True)
with col2:
	st.image("imgs/h2.jpg", use_column_width=True)
with col3:
	st.image("imgs/h3.jpg", use_column_width=True)


st.caption('A tall building with lots of signs on it.標識がたくさんある高層ビル。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/i1.jpg", use_column_width=True)
with col2:
	st.image("imgs/i2.jpg", use_column_width=True)
with col3:
	st.image("imgs/i3.jpg", use_column_width=True)


st.caption('A building with a lot of windows and a ceiling fan.窓がたくさんあり、天井ファンがたくさんある建物。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/j1.jpg", use_column_width=True)
with col2:
	st.image("imgs/j2.jpg", use_column_width=True)
with col3:
	st.image("imgs/j3.jpg", use_column_width=True)

	
st.caption('A city street filled with lots of traffic.多くの交通で満たされた街の通り。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/k1.jpg", use_column_width=True)
with col2:
	st.image("imgs/k2.jpg", use_column_width=True)
with col3:
	st.image("imgs/k3.jpg", use_column_width=True)


st.caption('A train is traveling down the tracks in a city.列車が都市の線路を下っています。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/l1.jpg", use_column_width=True)
with col2:
	st.image("imgs/l2.jpg", use_column_width=True)
with col3:
	st.image("imgs/l3.jpg", use_column_width=True)

	
st.caption('A large building with a clock tower in the background.背景に時計塔がある大きな建物。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/m1.jpg", use_column_width=True)
with col2:
	st.image("imgs/m2.jpg", use_column_width=True)
with col3:
	st.image("imgs/m3.jpg", use_column_width=True)


st.caption('A motorcycle parked on a street next to a building.建物の隣の通りに駐車したオートバイ。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/o1.jpg", use_column_width=True)
with col2:
	st.image("imgs/o2.jpg", use_column_width=True)
with col3:
	st.image("imgs/o3.jpg", use_column_width=True)
	

st.caption('A close up of a person wearing a hooded jacket.フード付きジャケットを着ている人のクローズアップ。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/p1.jpg", use_column_width=True)
with col2:
	st.image("imgs/p2.jpg", use_column_width=True)
with col3:
	st.image("imgs/p3.jpg", use_column_width=True)

	
st.caption('A black and white photo of a woman with a tie.ネクタイのある女性の白黒写真。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/q1.jpg", use_column_width=True)
with col2:
	st.image("imgs/q2.jpg", use_column_width=True)
with col3:
	st.image("imgs/q3.jpg", use_column_width=True)


st.caption('A bicycle parked on the side of a road next to a dog.犬の隣の道路の脇に駐車されている自転車。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/r1.jpg", use_column_width=True)
with col2:
	st.image("imgs/r2.jpg", use_column_width=True)
with col3:
	st.image("imgs/r3.jpg", use_column_width=True)
	

st.caption('Three small teddy bears sitting on a bench.ベンチに座っている3つの小さなテディベア。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/s1.jpg", use_column_width=True)
with col2:
	st.image("imgs/s2.jpg", use_column_width=True)
with col3:
	st.image("imgs/s3.jpg", use_column_width=True)

	
st.caption('A street with a train on the tracks and a building.線路と建物に電車がある通り。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/t1.jpg", use_column_width=True)
with col2:
	st.image("imgs/t2.jpg", use_column_width=True)
with col3:
	st.image("imgs/t3.jpg", use_column_width=True)


st.caption('A woman walking down a street while holding an umbrella.傘を持って通りを歩いている女性。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/u1.jpg", use_column_width=True)
with col2:
	st.image("imgs/u2.jpg", use_column_width=True)
with col3:
	st.image("imgs/u3.jpg", use_column_width=True)
	
	

st.caption('A group of people standing around a skateboard.スケートボードの周りに立っている人々のグループ。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/v1.jpg", use_column_width=True)
with col2:
	st.image("imgs/v2.jpg", use_column_width=True)
with col3:
	st.image("imgs/v3.jpg", use_column_width=True)

	
st.caption('A dog is standing on a brick sidewalk.犬がレンガの歩道に立っています。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/w1.jpg", use_column_width=True)
with col2:
	st.image("imgs/w2.jpg", use_column_width=True)
with col3:
	st.image("imgs/w3.jpg", use_column_width=True)


st.caption('A group of women standing next to each other.隣同士の女性のグループ。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/x1.jpg", use_column_width=True)
with col2:
	st.image("imgs/x2.jpg", use_column_width=True)
with col3:
	st.image("imgs/x3.jpg", use_column_width=True)
	

st.caption('A couple of people in a bathtub with mountains in the background.背景に山がある浴槽にいる数人の人々。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/y1.jpg", use_column_width=True)
with col2:
	st.image("imgs/y2.jpg", use_column_width=True)
with col3:
	st.image("imgs/y3.jpg", use_column_width=True)

	
st.caption('A boy in a black jacket holding a baseball bat.野球のバットを持っている黒いジャケットを着た男の子。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/z1.jpg", use_column_width=True)
with col2:
	st.image("imgs/z2.jpg", use_column_width=True)
with col3:
	st.image("imgs/z3.jpg", use_column_width=True)


st.caption('A person walking down a path with a dog.犬と一緒に道を歩いている人。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/aa1.jpg", use_column_width=True)
with col2:
	st.image("imgs/aa2.jpg", use_column_width=True)
with col3:
	st.image("imgs/aa3.jpg", use_column_width=True)
	

st.caption('A woman holding a baseball bat in her hands.野球のバットを手に持っている女性。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/bb1.jpg", use_column_width=True)
with col2:
	st.image("imgs/bb2.jpg", use_column_width=True)
with col3:
	st.image("imgs/bb3.jpg", use_column_width=True)

	
st.caption('A train yard with several train tracks and a train on one of them.いくつかの列車の線路とそのうちの1つに列車がある列車ヤード。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/cc1.jpg", use_column_width=True)
with col2:
	st.image("imgs/cc2.jpg", use_column_width=True)
with col3:
	st.image("imgs/cc3.jpg", use_column_width=True)


st.caption('A river with rocks and a waterfall running through it.岩と滝のある川が流れています。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/dd1.jpg", use_column_width=True)
with col2:
	st.image("imgs/dd2.jpg", use_column_width=True)
with col3:
	st.image("imgs/dd3.jpg", use_column_width=True)
	

st.caption('A man walking down a street past a tall building.背の高い建物を過ぎて通りを歩いている男。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/ee1.jpg", use_column_width=True)
with col2:
	st.image("imgs/ee2.jpg", use_column_width=True)
with col3:
	st.image("imgs/ee3.jpg", use_column_width=True)
	
st.caption('A group of men in uniform with a dog.犬と一緒に制服を着た男性のグループ。')
col1, col2, col3 = st.columns(3)
with col1:
	st.image("imgs/ff1.jpg", use_column_width=True)
with col2:
	st.image("imgs/ff2.jpg", use_column_width=True)
with col3:
	st.image("imgs/ff3.jpg", use_column_width=True)
	
	
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

