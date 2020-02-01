import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv2
import glob

# from mathutils import geometry as pygeo
import mathutils
from mathutils import Vector
import json

from mpl_toolkits import mplot3d

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('chessboards/*.png')
i = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret1, corners = cv2.findChessboardCorners(gray, (7,7), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if ret1:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,7), corners2, ret1)
        plt.imshow(img)
        name = 'img' + str(i) + '.png'
        cv2.imwrite(name, img)
        plt.show()
    i += 1

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)


for fname in glob.glob('chessboards/c2*.png'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,7),None)
    if ret:
        print(fname)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret, pnprvecs, pnptvecs = cv2.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, pnprvecs, pnptvecs, mtx, dist)
        img = draw(img,corners2,imgpts)
        plt.imshow(img)
        plt.show()

# Rotation matrix
rmatRight = cv2.Rodrigues(rvecs[0])[0]
rmatLeft = cv2.Rodrigues(rvecs[4])[0]

# Translation matrix
rotMatRight = np.concatenate((rmatRight, tvecs[0]), axis=1)
rotMatLeft = np.concatenate((rmatLeft, tvecs[4]), axis=1)

# Camera matrix
camLeft = mtx @ rotMatLeft
camRight = mtx @ rotMatRight

# Center of projection matrix
camWorldCenterLeft = np.linalg.inv(np.concatenate((rotMatLeft,[[0,0,0,1]]), axis=0)) @ np.transpose([[0,0,0,1]])
camWorldCenterRight = np.linalg.inv(np.concatenate((rotMatRight,[[0,0,0,1]]), axis=0)) @ np.transpose([[0,0,0,1]])

# fmat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
#
# s1 = fmat @ rmatRight
# s2 = fmat @ rmatLeft


def plotDotWorld():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(objp[:, 0], objp[:, 1], objp[:, 2])

    x, y, z, d = camWorldCenterLeft
    ax.scatter(x, y, z, c='g', marker='o')

    x2, y2, z2, d2 = camWorldCenterRight
    ax.scatter(x2, y2, z2, c='g', marker='o')

    fig.show()


plotDotWorld()


def crossMat(v):
    v = v[:, 0]
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def matFondamental(camLeft, centerRight, camRight):
    return np.array(crossMat(camLeft @ centerRight) @ camLeft @ np.linalg.pinv(camRight))


def getRed(fname):
    img = cv2.imread(fname)
    red = img[:, :, 2]
    ret, mask = cv2.threshold(red, 127, 255, cv2.THRESH_TOZERO)
    return mask


def getEpiLines(F, points):
    return F @ points


def findEpilines(path):
    epilines = []

    for l in range(26):
        if l < 10:
            strp = path + '000' + str(l) + '.png'
        else:
            strp = path + '00' + str(l) + '.png'

        red = getRed(strp)
        tempEpilines = []
        pointsLeft = [[], [], []]

        for i, line in enumerate(red):
            for pixel in line:
                if pixel != 0:
                    pixel = 1
            try:
                pointsLeft[0].append(np.average(range(1920), weights=line))
                pointsLeft[1].append(i)
                pointsLeft[2].append(1)
            except:
                pass

        epilinesRight = getEpiLines(Fondamental, pointsLeft)
        tempEpilines.append(pointsLeft)
        tempEpilines.append(epilinesRight)
        epilines.append(tempEpilines)
    return epilines


Fondamental = matFondamental(camRight, camWorldCenterLeft, camLeft)
# epl = [ [ [Red_x_avg], [Y_avg], [1] ], [EpilineRight(i)] ] ]
epl = findEpilines('scanLeft/')


def drawAvgPoint(fname, EplLeft):
    img = cv2.imread(fname)
    i = 0
    while i < len(EplLeft[0]):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img = cv2.circle(img, (int(EplLeft[0][i]), int(EplLeft[1][i])), 5, color, -1)
        i += 10
    plt.imshow(img)
    plt.show()


def lineY(coef, x):
    a, b, c = coef
    return -(c + a * x) / b


def drawEpl(fname, EplRight):
    img = cv2.imread(fname)
    coef, length = EplRight.shape
    for i in range(0, length, 40):
        print(EplRight[:, i])
        plt.plot([0, 1919], [lineY(EplRight[:, i], 0), lineY(EplRight[:, i], 1919)], 'r')

    plt.imshow(img)
    plt.show()


drawAvgPoint('scanLeft/0013.png', epl[13][0])
drawEpl('scanRight/scan0013.png', epl[13][1])


def getReddAvg(fname):
    red = getRed(fname)
    redPoints = [[], [], []]

    for i, line in enumerate(red):
        for pixel in line:
            if pixel != 0:
                pixel = 1
        try:
            redPoints[0].append(np.average(range(1920), weights=line))
            redPoints[1].append(i)
            redPoints[2].append(1)
        except:
            pass
    return redPoints


def eplRedPoints(path, EplRight):
    points = []
    for l in range(26):
        if l < 10:
            strp = path + '000' + str(l) + '.png'
        else:
            strp = path + '00' + str(l) + '.png'

        redPoints = getReddAvg(strp)
        scan = cv2.imread(strp)

        pointsRight = [[], [], []]
        eplImg = EplRight[l][1]
        print(strp)
        for i in range(len(eplImg[0])):
            try:
                x = int(redPoints[0][i])
                y = int(lineY(eplImg[:, i], x))
                pointsRight[0].append(x)
                pointsRight[1].append(y)
                pointsRight[2].append(1)

                color = tuple(np.random.randint(0, 255, 3).tolist())
                scan = cv2.circle(scan, (x, y), 5, color, -1)
            except:
                pass
        points.append(pointsRight)
        # plt.imshow(scan)
        # plt.show()
    return points


pointsRight = eplRedPoints('scanRight/scan', epl)

def arrayToVector(p):
    return Vector((p[0], p[1], p[2]))


def getIntersection(pointsLeft, pointsRight):
    pL = np.array(pointsLeft)
    pR = np.array(pointsRight)

    camCenterRight = np.transpose(camWorldCenterRight)[0]
    camCenterLeft = np.transpose(camWorldCenterLeft)[0]

    # calcul du point sur l'object en applicant la pseudo-inverse de la camera sur le point trouvé plus-haut

    leftObject = (np.linalg.pinv(camLeft) @ pL)
    rightObject = (np.linalg.pinv(camRight) @ pR)

    # conversion des np.array en mathutils.Vector pour l'utilisation de la methode d'intersection

    leftEndVec = arrayToVector(leftObject)
    rightEndVec = arrayToVector(rightObject)

    leftStartVec = arrayToVector(camCenterLeft)
    rightStartVec = arrayToVector(camCenterRight)

    # affichage des lignes reliant centre à point objet

    '''
    draw3DLine(camCenterLeft,leftObject)
    draw3DLine(camCenterRight,rightObject)
    plt.show()
    '''

    # utilisation de mathutils.geometry.intersect_line_line pour trouver l'intersection des lingnes passant par les 2
    # points.
    return mathutils.geometrie.intersect_line_line(leftStartVec, leftEndVec, rightStartVec, rightEndVec)


def draw3DLine(start, end):
    figure = plt.figure()
    ax = Axes3D(figure)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    x_start, y_start, z_start = start
    x_end, y_end, z_end = end

    print("start = ({},{},{})".format(x_start, y_start, z_start))
    print("end = ({},{},{})\n".format(x_end, y_end, z_end))

    ax.scatter(x_start, y_start, z_start, c='r', marker='o')
    ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end])


def getObjectPoint():
    point = [[], [], []]
    for l in range(26):
        pointsLeft = np.array(epl[l][0])

        pointRight = np.array(pointsRight[l])
        for i in range(len(pointsLeft[0])):
            try:

                # calcul du point d'intersection sur l'objet -> on obtient une liste de vector
                intersection = getIntersection(pointsLeft[:, i], pointRight[:, i])
                # print(intersection)
                for inter in intersection:
                    inter *= 1000
                    x, y, z = inter
                    point[0].append(x)
                    point[1].append(y)
                    point[2].append(z)
            except:
                pass
    return np.array(point)


def drawPointObject(point):
    figure = plt.figure()
    ax = Axes3D(figure)

    ax.scatter3D(point[0, :], point[1, :], point[2, :], c='black', marker='x')

    ax.view_init(-95, -50)
    plt.axis('off')
    plt.show()


def drawSurfaceObject(point):
    figure = plt.figure()
    ax = Axes3D(figure)
    ax.plot_trisurf(point[0, :], point[1, :], point[2, :])

    ax.view_init(-95, -50)
    plt.axis('off')
    plt.show()


def pointToJson(point):
    data = {'x': point[0, :].tolist(), 'y': point[1, :].tolist(), 'z': point[2, :].tolist()}
    with open('point.txt', '+w') as file:
        json.dump(data, file)


point = getObjectPoint()
pointToJson(point)
drawSurfaceObject(point)
drawPointObject(point)


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(objp[:,0],objp[:,1],0)
# plt.show()

# img1 = cv2.imread('chessboards/c2Left.png',0)  #queryimage # left image
# img2 = cv2.imread('chessboards/c2Right.png',0) #trainimage # right image

# sift = cv2.SIFT()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
#
# # FLANN parameters
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)
#
# flann = cv2.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des1,des2,k=2)
#
# good = []
# pts1 = []
# pts2 = []
#
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.8*n.distance:
#         good.append(m)
#         pts2.append(kp2[m.trainIdx].pt)
#         pts1.append(kp1[m.queryIdx].pt)
#
# pts1 = np.int32(pts1)
# pts2 = np.int32(pts2)
# F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
#
# # We select only inlier points
# pts1 = pts1[mask.ravel()==1]
# pts2 = pts2[mask.ravel()==1]
#
#
# def drawlines(img1,img2,lines,pts1,pts2):
#     ''' img1 - image on which we draw the epilines for the points in img2
#         lines - corresponding epilines '''
#     r,c = img1.shape
#     img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#     img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
#     for r,pt1,pt2 in zip(lines,pts1,pts2):
#         color = tuple(np.random.randint(0,255,3).tolist())
#         x0,y0 = map(int, [0, -r[2]/r[1] ])
#         x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#         img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
#         img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
#         img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
#     return img1,img2
#
#
# # Find epilines corresponding to points in right image (second image) and
# # drawing its lines on left image
# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
#
# # Find epilines corresponding to points in left image (first image) and
# # drawing its lines on right image
# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
#
# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()

