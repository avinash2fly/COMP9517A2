from __future__ import print_function  #
import cv2
import argparse
import os
import imghdr
from numpy import linalg as LA
import numpy as np
import sys
import random

imgformat=['jpeg']
surf = cv2.xfeatures2d.SURF_create(400)
# surf.setUpright(True)
surf.setExtended(True)
surf.setNOctaves(4)
surf.setNOctaveLayers(2)

class MyImage:
    def __init__(self, img_name):
        self.img = None
        self.__name = img_name

    def __str__(self):
        return self.__name


def checkdir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

def getSURFFeatures(im):
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		kp, des = surf.detectAndCompute(gray, None)
		return {'kp':kp, 'des':des}

def up_to_step_1(imgs):
    """Complete pipeline up to step 3: Detecting features and descriptors"""
    # ... your code here ...
    newImg =[]
    for img1 in imgs:
        img = img1.img
        features = getSURFFeatures(img)
        out = cv2.drawKeypoints(img,features['kp'],None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        newImg.append(out)

    return newImg


def save_step_1(imgs, output_path='./output/step1'):
    """Save the intermediate result from Step 1"""
    # # ... your code here ...
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    checkdir(output_path)
    count=1
    for img in imgs:
        filename = output_path + '/output' +str(count)+'.jpg'
        cv2.imwrite(filename,img)
        count =count+1

def getMatches(F_des1,F_des2,thres = 0.6):
    matches=[];
    for idx1,des1 in enumerate(F_des1):
        nearestDistance = [np.inf,np.inf]
        nearestPoints = [-1,-1]
        for idx2,des2 in enumerate(F_des2):
            dist = LA.norm(des1-des2)
            if dist < nearestDistance[0]:
                nearestDistance = [dist] + nearestDistance[:-1]
                nearestPoints = [idx2] + nearestPoints[:-1]
            elif dist < nearestDistance[1]:
                nearestDistance[1] = dist
                nearestPoints[1] = idx2
        if nearestDistance[0]/nearestDistance[1] > thres:
            continue
        matches.append([cv2.DMatch(idx1,nearestPoints[0],nearestDistance[0])])
    return np.array(matches)

def up_to_step_2(imgs):
    """Complete pipeline up to step 2: Calculate matching feature points"""
    # ... your code here ...

    modified_imgs=[]
    final_matches=[]
    for i in range(0,len(imgs)):
        for j in range(i+1,len(imgs)):
            features1 = getSURFFeatures(imgs[i].img)
            features2 = getSURFFeatures(imgs[j].img)
            matches=getMatches(features1['des'],features2['des'])
            modified_imgs.append([imgs[i],features1,imgs[j],features2])
            final_matches.append(matches)
            # img3 = cv2.drawMatchesKnn(imgs[0], features1['kp'], imgs[1], features2['kp'], matches,None,
            #                   matchColor = (0,255,255), singlePointColor = (255,0,0),flags=2)
    # cv2.imwrite('/Users/avinashgupta/PycharmProjects/COMP9517A2/output/out.jpg',img3)
    return modified_imgs, final_matches


def save_step_2(imgs, match_list, output_path="./output/step2"):
    """Save the intermediate result from Step 2"""
    # ... your code here ...
    checkdir(output_path)
    for index in range(len(imgs)):
        imgrow = imgs[index]
        img1 = imgrow[0]
        feature1 = imgrow[1]
        img2 = imgrow[2]
        feature2 = imgrow[3]
        matches = match_list[index]
        img3 = cv2.drawMatchesKnn(img1.img, feature1['kp'], img2.img, feature2['kp'], matches, None,
                          matchColor = (0,255,255), singlePointColor = (255,0,0),flags=2)
        filename = output_path + '/output' + str(img1)+'_'+str(len(feature1['kp'])) +str(img2)+'_'+str(len(feature2['kp']))+'_'+str(len(matches))+ '.jpg'
        cv2.imwrite(filename,img3)



def calculateHomography(correspondences):
    #loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    #svd composition
    u, s, v = np.linalg.svd(matrixA)

    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    #normalize and now we have h
    h = (1/h.item(8)) * h
    return h


#
#Calculate the geometric distance between estimated points and original points
#
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


#
#Runs through ransac algorithm, creating homographies from random correspondences
#
def ransac(corr, thresh):
    maxInliers = []
    finalH = None
    for i in range(1000):
        #find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        #call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h

        if len(maxInliers) > (len(corr)*thresh):
            break
    return np.array(finalH), maxInliers

def wrapImage(self, leftImage, warpedImage):
		i1y, i1x = leftImage.shape[:2]
		i2y, i2x = warpedImage.shape[:2]

		black_l = np.where(leftImage == np.array([0,0,0]))
		black_wi = np.where(warpedImage == np.array([0,0,0]))

		for i in range(0, i1x):
			for j in range(0, i1y):
				try:
					if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
						# print "BLACK"
						# instead of just putting it with black,
						# take average of all nearby values and avg it.
						warpedImage[j,i] = [0, 0, 0]
					else:
						if(np.array_equal(warpedImage[j,i],[0,0,0])):
							# print "PIXEL"
							warpedImage[j,i] = leftImage[j,i]
						else:
							if not np.array_equal(leftImage[j,i], [0,0,0]):
								bw, gw, rw = warpedImage[j,i]
								bl,gl,rl = leftImage[j,i]
								# b = (bl+bw)/2
								# g = (gl+gw)/2
								# r = (rl+rw)/2
								warpedImage[j, i] = [bl,gl,rl]
				except:
					pass

		return warpedImage


def wrapImage(img,H,shape):
    h, w = shape
    indy, indx = np.indices((h, w), dtype=np.float32)
    lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])

    # warp the coordinates of src to those of true_dst
    map_ind = H.dot(lin_homg_ind)
    map_x, map_y = map_ind[:-1] / map_ind[-1]  # ensure homogeneity
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)
    newImg = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return newImg

def leftPerspective(a,b,H):
    xh = LA.inv(H)
    return righPerspective(a,b,xh)
    # txyz = np.dot(xh, np.array([b.shape[1], b.shape[0], 1]))
    # txyz = txyz / txyz[-1]
    # dsize = (int(txyz[0]) + a.shape[1], int(txyz[1]) + a.shape[0])
    # wrapedImage = wrapImage(a, xh, dsize)
    # return wrapedImage

def righPerspective(a,b,H):
    txyz = np.dot(H, np.array([b.shape[1], b.shape[0], 1]))
    txyz = txyz / txyz[-1]
    dsize = (int(txyz[0]) + a.shape[1], int(txyz[1]) + a.shape[0])
    wrapedImage = wrapImage(a, H, dsize)
    return wrapedImage

def set3Name(left,right):
    return left + "_"+ right +".jpg"

def up_to_step_3(imgs):
    """Complete pipeline up to step 3: estimating homographies and warpings"""
    # ... your code here ...
    newImage=[]
    modified_imgs, final_matches = up_to_step_2(imgs)
    for index in range(len(modified_imgs)):
        imgrow = modified_imgs[index]
        img1 = imgrow[0]
        feature1 = imgrow[1]
        img2 = imgrow[2]
        feature2 = imgrow[3]
        matches = final_matches[index]
        correspondenceList = []
        for match in matches:
            (x1, y1) = feature1['kp'][match[0].queryIdx].pt
            (x2, y2) = feature2['kp'][match[0].trainIdx].pt
            correspondenceList.append([x1, y1, x2, y2])
        corrs = np.matrix(correspondenceList)
        H, inliers = ransac(corrs, 0.6)

        leftImage = leftPerspective(img1.img,img2.img,H)
        rightImage = righPerspective(img2.img, img1.img, H)
        # cv2.imwrite("/Users/avinashgupta/PycharmProjects/COMP9517A2/output/out1.jpg",leftImage)
        # cv2.imwrite("/Users/avinashgupta/PycharmProjects/COMP9517A2/output/out2.jpg", rightImage)
        leftName = set3Name(str(img1).split('.')[0],str(img2).split('.')[0])
        rightName = set3Name(str(img2).split('.')[0], str(img1).split('.')[0])
        newImage.append([leftName,leftImage,rightName,rightImage])
    return newImage


def save_step_3(img_pairs, output_path="./output/step3"):
    """Save the intermediate result from Step 3"""
    # ... your code here ...
    checkdir(output_path)
    for img in img_pairs:
        cv2.imwrite(output_path+'/'+img[0],img[1])
        cv2.imwrite(output_path+'/'+img[2], img[3])


def up_to_step_4(imgs):
    """Complete the pipeline and generate a panoramic image"""
    # ... your code here ...
    return imgs[0]


def save_step_4(imgs, output_path="./output/step4"):
    """Save the intermediate result from Step 4"""
    # ... your code here ...
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        help="compute image stitching pipeline up to this step",
        type=int
    )

    parser.add_argument(
        "input",
        help="a folder to read in the input images",
        type=str
    )

    parser.add_argument(
        "output",
        help="a folder to save the outputs",
        type=str
    )

    args = parser.parse_args()

    imgs = []
    filelist = os.listdir(args.input)
    filelist.sort()
    for filename in filelist[:4]:
        # img = cv2.imread(os.path.join(args.input, filename))
        img = MyImage(filename)
        img.img = cv2.imread(os.path.join(args.input, filename))
        if img.img is None:
            continue
        print(filename)
        imgs.append(img)

    if args.step == 1:
        print("Running step 1")
        modified_imgs = up_to_step_1(imgs)
        save_step_1(modified_imgs, args.output)
    elif args.step == 2:
        print("Running step 2")
        modified_imgs, match_list = up_to_step_2(imgs)
        save_step_2(modified_imgs, match_list, args.output)
    elif args.step == 3:
        print("Running step 3")
        img_pairs = up_to_step_3(imgs)
        save_step_3(img_pairs, args.output)
    elif args.step == 4:
        print("Running step 4")
        panoramic_img = up_to_step_4(imgs)
        save_step_4(img_pairs, args.output)
