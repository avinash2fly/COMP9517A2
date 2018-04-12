from __future__ import print_function  #
import cv2
import argparse
import os
import imghdr
from numpy import linalg as LA
import numpy as np
import sys
import random
import math

imgformat=['jpeg']
surf = cv2.xfeatures2d.SURF_create(400,1,1,True,False)
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

def resize(img,ratio=0.1):
    return cv2.resize(img,(0,0),fx=ratio,fy=ratio)

def checkdir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

def getSURFFeatures(im):
		gray = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),(5,5), 0)
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
    try:
        estimatep2 = (1/estimatep2.item(2))*estimatep2
    except:
        pass

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


def wrapImage(img, H, shape=None, resize=False):
    row, col = shape
    y_cor, x_cor = np.indices((row, col), dtype=np.float32)
    idx1 = np.array([x_cor.ravel(), y_cor.ravel(), np.ones_like(x_cor).ravel()])
    idx2 = np.linalg.inv(H).dot(idx1)

    # warp(img1.img, H1, img1.img.shape[:2])
    if resize:
        # Calculate the size of the destination image
        xmin, ymin = np.min(idx2[:-1] / idx2[-1], axis=1)
        xmax, ymax = np.max(idx2[:-1] / idx2[-1], axis=1)
        xoffset = int(np.floor(min(xmin, 0)))
        yoffset = int(np.floor(min(ymin, 0)))
        H_trans = np.array([[1, 0, xoffset], [0, 1, yoffset], [0, 0, 1]])
        col, row = max(int(np.ceil(xmax - xmin)), img.shape[0]), max(int(round(ymax - ymin)), img.shape[1])

        y_cor, x_cor = np.indices((row, col), dtype=np.float32)
        idx1 = np.array([x_cor.ravel(), y_cor.ravel(), np.ones_like(x_cor).ravel()])
        idx2 = H_trans.dot(np.linalg.inv(H).dot(idx1))

    map_x, map_y = idx2[:-1] / idx2[-1]
    map_x = map_x.reshape(row, col).astype(np.float32)
    map_y = map_y.reshape(row, col).astype(np.float32)

    newImg = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return newImg

def leftPerspective(a,b,H):
    xh = LA.inv(H)
    return righPerspective(a,b,xh)

def righPerspective(a,b,H):
    wrapedImage = wrapImage(a, H, a.shape[:2],True)
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

def match(a,b,direction):
    imageSet1 = getSURFFeatures(a)
    imageSet2 = getSURFFeatures(b)
    matches = getMatches(imageSet1['des'],imageSet2['des'])
    correspondenceList = []
    for match in matches:
        (x1, y1) = imageSet1['kp'][match[0].queryIdx].pt
        (x2, y2) = imageSet2['kp'][match[0].trainIdx].pt
        correspondenceList.append([x1, y1, x2, y2])
    corrs = np.matrix(correspondenceList)
    H, inliers = ransac(corrs, 0.6)
    return H, inliers

# def stitch(imgs):
#
# 	prepare_lists()

def prepare_lists(images):
    count = len(images)
    centerIdx = count//2
    left_list, right_list, center_im = [], [], None
    center_im = images[centerIdx]
    for i in range(count):
        if(i<=centerIdx):
            left_list.append(images[i])
        else:
            right_list.append(images[i])
    return left_list,right_list



def findDimensions(image, homography):
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)
    (y, x) = image.shape[:2]
    base_p1[:2] = [0, 0]
    base_p2[:2] = [x, 0]
    base_p3[:2] = [0, y]
    base_p4[:2] = [x, y]
    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:

        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

        hp_arr = np.array(hp, np.float32)

        normal_pt = np.array([hp_arr[0] / hp_arr[2], hp_arr[1] / hp_arr[2]], np.float32)

        if (max_x == None or normal_pt[0, 0] > max_x):
            max_x = normal_pt[0, 0]

        if (max_y == None or normal_pt[1, 0] > max_y):
            max_y = normal_pt[1, 0]

        if (min_x == None or normal_pt[0, 0] < min_x):
            min_x = normal_pt[0, 0]

        if (min_y == None or normal_pt[1, 0] < min_y):
            min_y = normal_pt[1, 0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return (min_x, min_y, max_x, max_y)

def up_to_step_4(imgs):
    """Complete the pipeline and generate a panoramic image"""
    # ... your code here ...

    closestImage = None
    base_img_rgb = imgs[0].img
    base_img = cv2.GaussianBlur(cv2.cvtColor(base_img_rgb, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    for temp in imgs[1:]:
        next_img_rgb = temp.img
        next_img = cv2.GaussianBlur(cv2.cvtColor(next_img_rgb, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        H, status = match(base_img_rgb, next_img_rgb,'')
        inlierRatio = float(np.sum(status)) / float(len(status))

        if (closestImage == None or inlierRatio > closestImage['inliers']):
            closestImage = {}
            closestImage['h'] = H
            closestImage['inliers'] = inlierRatio
            closestImage['rgb'] = next_img_rgb
            closestImage['img'] = next_img

    H = closestImage['h']
    H = H / H[2, 2]
    H_inv = LA.inv(H)

    if (closestImage['inliers'] > 0.1):  # and

        (min_x, min_y, max_x, max_y) = findDimensions(closestImage['img'], H_inv)

        # Adjust max_x and max_y by base img size
        max_x = max(max_x, base_img.shape[1])
        max_y = max(max_y, base_img.shape[0])

        move_h = np.matrix(np.identity(3), np.float32)

        if (min_x < 0):
            move_h[0, 2] += -min_x
            max_x += -min_x

        if (min_y < 0):
            move_h[1, 2] += -min_y
            max_y += -min_y

        mod_inv_h = move_h * H_inv

        img_w = int(math.ceil(max_x))
        img_h = int(math.ceil(max_y))


        # Warp the new image given the homography from the old image
        base_img_warp = wrapImage(base_img_rgb, move_h, (img_w, img_h))


        next_img_warp = wrapImage(closestImage['rgb'], mod_inv_h, (img_w, img_h))

        # Put the base image on an enlarged palette
        enlarged_base_img = np.zeros((img_w, img_h, 3), np.uint8)


        # Create a mask from the warped image for constructing masked composite
        (ret, data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY),
                                        0, 255, cv2.THRESH_BINARY)

        enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp,
                                    mask=np.bitwise_not(data_map),
                                    dtype=cv2.CV_8U)

        # Now add the warped image
        final_img = cv2.add(enlarged_base_img, next_img_warp,
                            dtype=cv2.CV_8U)


        # Crop off the black edges
        final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        max_area = 0
        best_rect = (0, 0, 0, 0)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # print "Bounding Rectangle: ", (x,y,w,h)

            deltaHeight = h - y
            deltaWidth = w - x

            area = deltaHeight * deltaWidth

            if (area > max_area and deltaHeight > 0 and deltaWidth > 0):
                max_area = area
                best_rect = (x, y, w, h)

        if max_area > 0:

            final_img_crop = final_img[best_rect[1]:best_rect[1] + best_rect[3],
                             best_rect[0]:best_rect[0] + best_rect[2]]


            final_img = final_img_crop

    return final_img


def save_step_4(img, output_path="./output/step4"):
    """Save the intermediate result from Step 4"""
    # ... your code here ...
    checkdir(output_path)
    filename = output_path + '/output.jpg'
    cv2.imwrite(filename, img)



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
    for filename in filelist[:50]:
        # img = cv2.imread(os.path.join(args.input, filename))
        img = MyImage(filename)
        temo = cv2.imread(os.path.join(args.input, filename))
        if temo is None:
            continue
        # img.img = temo
        img.img = resize(temo, 0.3)
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
        save_step_4(panoramic_img, args.output)
