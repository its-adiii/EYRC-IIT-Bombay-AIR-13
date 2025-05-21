# import the necessary packages

import numpy as np
import argparse
import imutils
import cv2

#----------------------------------------------------------------------------------------------------------

# load the image
parser = argparse.ArgumentParser()
parser.add_argument('--image',help='Path to the input image')
args = parser.parse_args()
img = cv2.imread(args.image)
txt_filename = (args.image).replace(".png",".txt")
img_filename = (args.image).replace(".png","_output.png")

#----------------------------------------------------------------------------------------------------------

#implementation logic updated for stage 2
erosionimg = cv2.erode(img, np.ones((7,7)), iterations=1)
grayimg = cv2.cvtColor(erosionimg, cv2.COLOR_BGR2GRAY)
_, thresholdimg = cv2.threshold(grayimg, 180, 255, cv2.THRESH_BINARY)
dilationimg = cv2.dilate(thresholdimg, np.ones((7,7)), iterations=1)

#--------------------------------------------------------------------------------------------------------------

# perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large" components

nos_Labels, label_id, stat , centroid = cv2.connectedComponentsWithStats(dilationimg, 4, 2)

#---------------------------------------------------------------------------------------------------------------

# find the contours in the mask, then sort them from left to right

contours = cv2.findContours(dilationimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

#----------------------------------------------------------------------------------------------

# Initialize lists to store centroid coordinates and area

centroid_list = []
area_list = []
centroids = []  
def centroid_calculator(centroid_list):
    alien = len(centroid_list)

    if alien in (2, 3, 4, 5):
        
        sumofx = sum(xcoor for xcoor, _ in centroid_list)
        sumofy = sum(ycoor for _, ycoor in centroid_list)
        centroidx = sumofx / alien
        centroidy = sumofy / alien
        return [['alien_' + chr(97 + alien - 2), alien, centroidx, centroidy]]

    elif alien >= 6:
        
        quadrants = [[] for _ in range(4)]
        for x, y in centroid_list:
            quadrant = 0 if x < 256 else 1 
            quadrant += 2 if y < 256 else 0 
            quadrants[quadrant].append((x, y))

        
        centroids = []
        for quad_list in quadrants:
            if quad_list:
                centroids.extend(centroid_calculator(quad_list))  

        return centroids

    else:
       
        return []

# Loop over the contours
for cont in contours:
    # Calculate the area of the contour
    area = cv2.contourArea(cont)
    # Calculate the centroid coordinates
    M = cv2.moments(cont)
    if M['m00'] != 0:
        cx = float(M['m10'] / M['m00'])
        cy = float(M['m01'] / M['m00'])
    else:
        # Handle the case when the area (m00) is zero to avoid division by zero
        cx, cy = 0, 0

    # Append centroid coordinates and area to the respective lists
    area_list.append(area)
    centroid_list.append((cx , cy))

# Draw the bright spot on the image
cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
#------------------------------------------------------------------------------------------------

# Open a text file for writing
with open(txt_filename, "w") as file:
    unique_centroids = set()  
    centroids = centroid_calculator(centroid_list)

    for alien, noofled, x, y in centroids:
        centroid_tuple = (alien, noofled, x, y)  
        if centroid_tuple not in unique_centroids:
            unique_centroids.add(centroid_tuple)
            file.write(f"Organism Type: {alien}\n" + f"Centroid: ({round(float(x), 2):.2f}, {round(float(y), 2):.2f})\n\n")

# Display the image with the drawn point
cv2.imwrite(img_filename, img)
