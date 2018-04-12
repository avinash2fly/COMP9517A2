from __future__ import print_function, division

import cv2
import argparse
import numpy as np
import os


# --------------------      Classes      -------------------- #

class image():
    def __init__(self, img):
        if type(img) == str:
            self.img = cv2.imread(img, 1)
        else:
            self.img = img

        self.keypoints = []
        self.descriptors = []

    def makeGrayscale(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def getKeypoints(self):
        detector = cv2.xfeatures2d.SURF_create(hessianThreshold=3000, nOctaves=4, nOctaveLayers=2,
                                               extended=False, upright=False)
        grayImg = cv2.GaussianBlur(self.makeGrayscale(), (5, 5), 0)
        self.keypoints, self.descriptors = detector.detectAndCompute(grayImg, None)

    def drawKeyPoints(self):
        if len(self.keypoints) == 0:
            self.getKeypoints()
        return cv2.drawKeypoints(self.img, self.keypoints, None, -1, 4)

    def resize(self, ratio=0.1):
        self.img = cv2.resize(self.img, (0, 0), fx=ratio, fy=ratio)


# -------------------- Helper functions  -------------------- #

def checkOutputDirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def SSD(ar1, ar2):
    return np.sqrt(sum((ar1 - ar2) ** 2))


def KNNmatcher(des1, des2, threshold=0.6):
    matches = []
    for i in range(len(des1)):
        p1 = des1[i]
        nearestDistance = [np.inf, np.inf]  # index 0 is for the nearest point, index 1 is the 2nd nearest
        nearestPoints = [-1, -1]

        for j in range(len(des2)):
            p2 = des2[j]
            distance = SSD(p1, p2)

            if distance < nearestDistance[0]:
                nearestDistance = [distance] + nearestDistance[:-1]
                nearestPoints = [j] + nearestPoints[:-1]
            elif distance < nearestDistance[1]:
                nearestDistance[1] = distance
                nearestPoints[1] = j

        if nearestDistance[0] / nearestDistance[1] > threshold:
            continue
        matches.append(cv2.DMatch(i, nearestPoints[0], nearestDistance[0]))

    return np.array(matches)


def computeH(kp1, kp2):
    '''
    Find H for the mapping kp2 = H*kp1
    '''

    ##    # For numerical stability
    ##    # What have I done here, I think i messed up something... This is what happened
    ##    # when you wiped up your memory and tried to do everything again in 2 days
    ##    m = np.mean(kp1, axis=0)
    ##    std = max(np.std(kp1, axis=0)) + 1e-9
    ##    C1 = np.diag([1/std, 1/std, 1])
    ##    C1[0][2] = -m[0]/std
    ##    C1[1][2] = -m[1]/std
    ##    kp1 = C1.dot(np.hstack((kp1,np.ones(shape=(kp1.shape[0],1)))).transpose())
    ##
    ##    m = np.mean(kp2, axis=0)
    ##    std = max(np.std(kp2, axis=0)) + 1e-9
    ##    C2 = np.diag([1/std, 1/std, 1])
    ##    C2[0][2] = -m[0]/std
    ##    C2[1][2] = -m[1]/std
    ##    kp2 = C2.dot(np.hstack((kp2,np.ones(shape=(kp2.shape[0],1)))).transpose())


    P = []
    for i in range(len(kp1)):
        P.append([-kp1[i][0], -kp1[i][1], -1, 0, 0, 0, kp1[i][0] * kp2[i][0], kp1[i][1] * kp2[i][0], kp2[i][0]])
        P.append([0, 0, 0, -kp1[i][0], -kp1[i][1], -1, kp1[i][0] * kp2[i][1], kp1[i][1] * kp2[i][1], kp2[i][1]])

    U, S, V = np.linalg.svd(P)
    H = V[-1].reshape((3, 3))

    # C2*kp2 =K*C1*kp1 so kp2 = inv(C2)*K*C1*kp1 such that kp2 = H*kp1, where H = inv(C2)*K*C1
    ##    H = np.linalg.inv(C2).dot(H.dot(C1))
    return H / H[2, 2]


def transformPt(pt, H):
    res = H.dot(np.array([pt[0], pt[1], 1]))
    return res[:-1] / res[-1]


def transform(pt, H):
    if len(pt.shape) > 1:
        arr = np.hstack((pt, np.ones(shape=(pt.shape[0], 1)))).transpose()
    else:
        arr = np.array([pt[0], pt[1], 1])
    res = H.dot(arr)
    return res[:-1] / res[-1]


def computeError(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=0))


def RANSAC(kp1, kp2, n_iter=100, n_samples=4, minInliers=10, maxError=10):
    errors = []
    bestModel = None
    bestError = np.inf
    bestIdx = None
    all_idx = [i for i in range(len(kp1))]
    i = 0

    while i < n_iter:
        testIdx = np.random.choice(all_idx, size=n_samples, replace=False)
        testPt1 = kp1[testIdx]
        testPt2 = kp2[testIdx]
        H = computeH(testPt1, testPt2)

        # Count inliers
        inliers = set()
        for j in all_idx:
            if j in testIdx:
                continue
            pt1 = kp1[j]
            pt2 = kp2[j]

            error = computeError(transform(pt1, H), pt2)
            if error < maxError:
                inliers.add(j)

        # Refit model using all inliers
        if len(inliers) > minInliers:
            inliersIdx = list(inliers.union(set(testIdx)))
            testPt1 = kp1[inliersIdx]
            testPt2 = kp2[inliersIdx]
            H = computeH(testPt1, testPt2)
            error = np.mean(computeError(transform(testPt1, H), testPt2.transpose()))
            errors.append(error)
            if error < bestError:
                bestModel = H.copy()
                bestError = error
                bestIdx = inliersIdx
        i += 1;

    return bestModel, bestError, bestIdx


def warp(img, H, shape=None, resize=True):
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


def testing(r=0.3):
    import time
    inputp = 'C:/Users/Yohanes/Desktop/ass2/test/'
    output = 'C:/Users/Yohanes/Desktop/ass2/out/'
    imgs = {}
    for filename in os.listdir(inputp):
        print(filename)
        img = image(inputp + filename)
        img.resize(ratio=r)
        imgs[filename] = img
    return imgs


# ------------------ Assignment functions  ------------------ #

def up_to_step_1(imgs):
    # Nothing to do here. drawKeyPoints() function is contained within the Image object class
    # It will be called in save_step_1
    return imgs


def save_step_1(imgs, output_path='./output/step1'):
    """Save the intermediate result from Step 1"""
    checkOutputDirectory(output_path)
    for img in imgs:
        cv2.imwrite(os.path.join(output_path, img), imgs[img].drawKeyPoints())


def up_to_step_2(imgs, minMatch=20):
    """Complete pipeline up to step 2: Calculate matching feature points"""
    matchedImages = []
    img_names = imgs.keys()
    for i in range(len(img_names) - 1):
        img1 = imgs[img_names[i]]
        img1.getKeypoints()
        for j in range(i + 1, len(img_names)):
            img2 = imgs[img_names[j]]
            img2.getKeypoints()
            matches = KNNmatcher(img1.descriptors, img2.descriptors)

            if len(matches) < minMatch:
                continue

            img12 = cv2.drawMatches(img1.img, img1.keypoints, img2.img, img2.keypoints, matches, None, None)
            matchedImages.append({"img1": img_names[i], "img2": img_names[j], "img12": img12, "matches": matches})

    return imgs, matchedImages


def save_step_2(imgs, match_list, output_path="./output/step2"):
    """Save the intermediate result from Step 2"""
    checkOutputDirectory(output_path)
    for match in match_list:
        kp_cnt1 = str(len(imgs[match["img1"]].keypoints))
        kp_cnt2 = str(len(imgs[match["img2"]].keypoints))
        pair = str(len(match["matches"]))
        img_name = '_'.join([match["img1"], kp_cnt1, match["img2"], kp_cnt2, pair]) + '.jpg'
        print(img_name)
        cv2.imwrite(os.path.join(output_path, img_name), match["img12"])


def up_to_step_3(imgs):
    """Complete pipeline up to step 3: estimating homographies and warpings"""
    _, match_list = up_to_step_2(imgs)

    warpedImgs = []

    for pair in match_list:
        img1 = imgs[pair["img1"]]
        img2 = imgs[pair["img2"]]
        kp1 = np.array([img1.keypoints[i.queryIdx].pt for i in pair["matches"]])
        kp2 = np.array([img2.keypoints[i.trainIdx].pt for i in pair["matches"]])
        H1, H2 = None, None
        while type(H1) != np.ndarray:
            H1, E, I = RANSAC(kp1, kp2)  # kp2 = H*kp1
        while type(H2) != np.ndarray:
            H2, E, I = RANSAC(kp2, kp1)  # kp1 = H*kp2

        warpedImgs.append(
            {"wimg": pair["img1"], "ref": pair["img2"], "H": H1, "warped": warp(img1.img, H1, img1.img.shape[:2])})
        warpedImgs.append(
            {"wimg": pair["img2"], "ref": pair["img1"], "H": H2, "warped": warp(img2.img, H2, img1.img.shape[:2])})

    return warpedImgs


def save_step_3(img_pairs, output_path="./output/step3"):
    """Save the intermediate result from Step 3"""
    checkOutputDirectory(output_path)
    for pair in img_pairs:
        img_name = '_'.join([pair["wimg"], pair["ref"]]) + '.jpg'
        print(img_name)
        cv2.imwrite(os.path.join(output_path, img_name), pair["warped"])


def getCorners(img, H):
    y, x = img.shape[:2]
    p = np.array([[0, x, 0, x],
                  [0, 0, y, y],
                  [1, 1, 1, 1]])
    Hp = H.dot(p)
    Jp = Hp[:2] / Hp[2, 2]
    xmax, ymax = np.max(Hp, axis=1)[:2]
    xmin, ymin = np.min(Hp, axis=1)[:2]
    xmin = min(xmin, 0)
    ymin = min(ymin, 0)
    return (xmin, xmax, ymin, ymax)


def up_to_step_4(imgs):
    """Complete the pipeline and generate a panoramic image"""
    pairs = {}
    if len(imgs) < 2:
        print('Need at least 2 images')
        return
    _, match_list = up_to_step_2(imgs)

    # Assuming left to right
    imgNames = imgs.keys()
    imgNames.sort()

    c = (len(imgNames) + 1) // 2 - 1
    if c == 0:
        group1 = []
    else:
        group1 = [k for k in range(c - 1, -1, -1)]
    group2 = [k for k in range(c + 1, len(imgNames))]

    # Start with the middle image
    img1 = imgs[imgNames[c]]

    cnt = 0
    for i in group1 + group2:
        # Find match
        img1.getKeypoints()
        img2 = imgs[imgNames[i]]
        img2.getKeypoints()
        matches = KNNmatcher(img1.descriptors, img2.descriptors, threshold=0.7)

        img12 = cv2.drawMatches(img1.img, img1.keypoints, img2.img, img2.keypoints, matches, None, None)

        # Get matching keypoints
        kp1 = np.array([img1.keypoints[m.queryIdx].pt for m in matches])
        kp2 = np.array([img2.keypoints[m.trainIdx].pt for m in matches])

        # Calculate homography
        H = None
        while type(H) != np.ndarray:
            H, E, I = RANSAC(kp2, kp1)  # kp1 = H*kp2

        (xmin, xmax, ymin, ymax) = getCorners(img2.img, H)
        xmax = max(xmax, img1.img.shape[0]) - xmin
        ymax = max(ymax, img1.img.shape[1]) - ymin

        H_trans = np.array([[1, 0, -xmin],
                            [0, 1, -ymin],
                            [0, 0, 1]])

        H_mod = H_trans.dot(H)
        row, col = int(np.ceil(xmax)), int(np.ceil(ymax))

        # Shift base image to make space and warp the second image
        img1_warped = warp(img1.img, H_trans, (row, col), resize=False)
        img2_warped = warp(img2.img, H_mod, (row, col), resize=False)

        # Create mask for merging later
        ret, mask = cv2.threshold(cv2.cvtColor(img1_warped, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)

        # Create a black canvas
        blank = np.zeros(shape=(row, col, 3), dtype=np.uint8)

        # Add the base image to the canvas with the previously created mask then add the second image.
        blank = cv2.add(blank, img2_warped, mask=np.bitwise_not(mask), dtype=cv2.CV_8U)
        combined = cv2.add(blank, img1_warped, dtype=cv2.CV_8U)

        # Trim excess black edges
        _, mask = cv2.threshold(cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        contour, _, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        max_area = 0
        best_rect = (0, 0, 0, 0)
        for c in contour:
            x1, y1, x2, y2 = cv2.boundingRect(c)
            h = y2 - y1
            w = x2 - x1
            area = h * w

            if area > max_area and h > 0 and w > 0:
                max_area = area
                best_rect = (x1, y1, x2, y2)

        if max_area > 0.5 * (row * col):
            cropped_combined = combined[best_rect[1]:best_rect[1] + best_rect[3]][
                               best_rect[0]:best_rect[0] + best_rect[2]]
            combined = cropped_combined.copy()

        # Set combined result as the new base
        img1 = image(combined)
        cnt += 1
    return img1


def save_step_4(imgs, output_path="./output/step4"):
    """Save the intermediate result from Step 4"""
    checkOutputDirectory(output_path)
    return cv2.imwrite(os.path.join(output_path, 'step4_output.jpg'), imgs.img)


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

    imgs = {}
    for filename in os.listdir(args.input):
        img = image(os.path.join(args.input, filename))
        # img.resize(ratio=0.3)
        imgs[filename] = img

    if args.step == 1:
        print("Running step 1")
        modified_imgs = up_to_step_1(imgs)
        save_step_1(imgs, args.output)
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

