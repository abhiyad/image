import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from Line import Line
from line_fit import line_fit, tune_fit, final_viz, calc_vehicle_offset
from moviepy.editor import VideoFileClip


# Global variables (just to make the moviepy video annotation work)
# with open('calibrate_camera.p', 'rb') as f:
# 	save_dict = pickle.load(f)
# mtx = save_dict['mtx']
# dist = save_dict['dist']

mtx = np.array([[1.15687796e+03, 0.00000000e+00, 6.70608746e+02],
       [0.00000000e+00, 1.15353388e+03, 3.88894697e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist = np.array([[-2.45563526e-01,  2.56430985e-03, -5.88014374e-04,
        -1.34979788e-04, -6.85209263e-02]])


window_size = 5  # how many frames for line smoothing
left_line = Line(n=window_size)
right_line = Line(n=window_size)
detected = False  # did the fast line fit detect the lines?
left_curve, right_curve = 0., 0.  # radius of curvature for left and right lanes
left_lane_inds, right_lane_inds = None, None  # for calculating curvature


# MoviePy video annotation will call this function
def annotate_image(img_in):
	"""
	Annotate the input image with lane line markings
	Returns annotated image
	"""
	global mtx, dist, left_line, right_line, detected
	global left_curve, right_curve, left_lane_inds, right_lane_inds

	# Undistort, threshold, perspective transform
	undist = cv2.undistort(img_in, mtx, dist, None, mtx)
	img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(undist)
	binary_warped, binary_unwarped, m, m_inv = perspective_transform(img)
	cv2.imwrite("image.jpg",binary_warped)
	# print("saved image")
	# print(np.unique(img))
	# print(np.unique(binary_warped))
	# Perform polynomial fit
	if not detected:
		# print "Line53"
		# Slow line fit
		ret = line_fit(binary_warped)
		left_fit = ret['left_fit']
		right_fit = ret['right_fit']
		nonzerox = ret['nonzerox']
		nonzeroy = ret['nonzeroy']
		left_lane_inds = ret['left_lane_inds']
		right_lane_inds = ret['right_lane_inds']

		# Get moving average of line fit coefficients
		left_fit = left_line.add_fit(left_fit)
		right_fit = right_line.add_fit(right_fit)
		# calculatingte curvature

		# print("HERE")
		#left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

		detected = True  # slow line fit always detects the line

	else:  # implies detected == True
		# Fast line fit
		# print "Line76"
		left_fit = left_line.get_fit()
		right_fit = right_line.get_fit()
		ret = tune_fit(binary_warped, left_fit, right_fit)
		left_fit = ret['left_fit']
		right_fit = ret['right_fit']
		nonzerox = ret['nonzerox']
		nonzeroy = ret['nonzeroy']
		left_lane_inds = ret['left_lane_inds']
		right_lane_inds = ret['right_lane_inds']

		# Only make updates if we detected lines in current frame
		if ret is not None:
			left_fit = ret['left_fit']
			right_fit = ret['right_fit']
			nonzerox = ret['nonzerox']
			nonzeroy = ret['nonzeroy']
			left_lane_inds = ret['left_lane_inds']
			right_lane_inds = ret['right_lane_inds']

			left_fit = left_line.add_fit(left_fit)
			right_fit = right_line.add_fit(right_fit)
			# print("HERE")
			#left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
		else:
			detected = False

	vehicle_offset = calc_vehicle_offset(undist, left_fit, right_fit)
	# print (vehicle_offset)
	# Perform final visualization on top of original undistorted image
	result = final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)

	return result


def annotate_video(input_file, output_file):
	""" Given input_file video, save annotated video to output_file """
	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(annotate_image)
	annotated_video.write_videofile(output_file, audio=False)

def advancedModel(img):
	return annotate_image(img)

if __name__ == '__main__':
	# Annotate the video
	#annotate_video('project_video.mp4', 'out.mp4')

	# Show example annotated image on screen for sanity check
	img_file = 'test_images/test2.jpg'
	img = mpimg.imread(img_file)
	# print(np.unique(img))
	# img=cv2.resize(img,(1280,720))
	img=cv2.resize(img,(640,360))
	# print(np.unique(img))
	t=time.time()
	result = annotate_image(img)
	print(1/(time.time()-t))
	plt.imshow(result)
	plt.show()
