import numpy as np
import pandas as pd
from skimage import io
import cv2 as cv
import matplotlib.pyplot as plt
import csv
import seaborn as sns

class microMSData():


	def __init__(self, coords_file_paths, image_path_dicts):


		self.coords_file_paths = coords_file_paths
		self.image_path_dicts = image_path_dicts
		self.channel_names = list(image_path_dicts[0].keys())


	def getCoords(self):

		for coords_file_path in self.coords_file_paths:
			coordinates, radius, names = self.parseCellCoords(coords_file_path)
			self.coords += coordinates
			self.radius += radius
			self.names += names


	def getCellImgs(self, crop_size, if_save=False, out_dir='./'):

		Coords = []
		Radius = []
		Names = []
		Cell_imgs = {}
		Integrated_intens = {}
		Mean_intens = {}
		for c in self.channel_names:
			Cell_imgs['cell_img_'+c] = []
			Mean_intens['mean_intens_'+c] = []
			Integrated_intens['integrated_intens_'+c] = []


		for coords_file_path, image_path_dict in zip(self.coords_file_paths,self.image_path_dicts):
			cell_imgs = {}
			for c in image_path_dict.keys():
				print('processing {} image'.format(c))
				cropped_imgs, integrated_intens, mean_intens, coord_list, radius, names = self.crop_cellimgs(coords_file_path, image_path_dict[c][0], crop_size, image_path_dict[c][1])
				Cell_imgs['cell_img_'+c] += cropped_imgs
				Integrated_intens['integrated_intens_'+c] += integrated_intens
				Mean_intens['mean_intens_'+c] += mean_intens
			Coords += coord_list
			Radius += radius
			Names += names

		self.obs = pd.concat([pd.DataFrame({'coordinates':Coords,'radius':Radius}), pd.DataFrame(Cell_imgs),
			pd.DataFrame(Integrated_intens), pd.DataFrame(Mean_intens)],axis=1)
		self.obs.index = Names

		del Cell_imgs



	def parseCellCoords(self, file_path):

	    with open(file_path) as fd:
	    	rd = csv.reader(fd, delimiter="\t", quotechar='"')
	    	rows = [row for row in rd]

	    coordinates = [[int(np.round(float(row[0]))),int(np.round(float(row[1])))] for row in rows[10:]]
	    radius = [float(row[2]) for row in rows[10:]]
	    names = ['x_{}y_{}'.format(coord[0],coord[1]) for coord in coordinates]

	    return coordinates, radius, names


	def crop_cellimgs(self, coords_file_path, img_path, crop_size, channel, if_save=False, out_dir=None):

		coord_list,radius,names = self.parseCellCoords(coords_file_path)
		print('parsing {} coordinates for cell locations...'.format(len(coord_list)))

		slide_img = io.imread(img_path)

		cropped_imgs = []
		img_names = []
		integrated_intens = []
		mean_intens = []

		idx = 0
		for coord,rad in zip(coord_list,radius):
		    img = slide_img[(coord[1]-crop_size):(coord[1]+crop_size), (coord[0]-crop_size):(coord[0]+crop_size), :]
		    mask = np.zeros((64,64), np.uint8)
		    mask = cv.circle(mask,(32,32),int(rad)+1,(255,255,255),-1)
		    img_masked = cv.bitwise_and(img,img,mask=mask)
		    integrated_intens.append(np.sum(img_masked,axis=(0,1))[channel])
		    mean_intens.append(cv.mean(img, mask=mask)[channel])
		    #img = (img - np.min(img))/np.ptp(img)

		    if if_save:
		        save_img = Image.fromarray(img)
		        save_img.save(out_dir+'/{}.tiff'.format(names[idx]))
		    cropped_imgs.append(img)
		    idx += 1


		del slide_img

		return cropped_imgs, integrated_intens, mean_intens, coord_list, radius, names


	def select_micromsIntens(self, channel1, channel2, thres1=(0,1.0,0.01), thres2=(0,1.0,0.01)):
    
		#fig = plt.figure(figsize=(4,4))

		x = np.array(self.obs['integrated_intens_'+channel1])
		y = np.array(self.obs['integrated_intens_'+channel2])

		self.obs['use'] = False
		use_index = (x>thres1*x.max()) &  (y>thres2*y.max())
		self.obs['use'].iloc[use_index] = True

		gs=sns.jointplot(x='integrated_intens_'+channel1,y='integrated_intens_'+channel2,
			hue='use',data=self.obs,edgecolors='k',s=10, height=4)
		gs.ax_joint.axvline(thres1*x.max(), color='r', linestyle='-')
		gs.ax_joint.axhline(thres2*y.max(), color='r', linestyle='-')


		#gs.scatterplot(x_selected,y_selected,color='red',ax=gs)

		gs.set_axis_labels(channel1.capitalize(),channel2.capitalize())

		plt.title("%s vs %s"%(channel1.capitalize(), channel2.capitalize()))

		plt.show()


	def show_cellImgs(self, name, figsize_h=1, figsize_w=3):
	
		cell_imgs = []
		for c in self.channel_names:
		    cell_imgs.append(self.obs.loc[name]['cell_img_'+c])

		fig,axes= plt.subplots(1,len(cell_imgs),figsize=(figsize_w,figsize_h))
		ax = axes.ravel()

		for i in range(len(cell_imgs)):
		    ax[i].imshow(cell_imgs[i])
		    ax[i].set_title(self.channel_names[i],fontsize=8)
		    ax[i].get_yaxis().set_visible(False)
		    ax[i].get_xaxis().set_visible(False)

		plt.show()



