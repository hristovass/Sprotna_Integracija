import math

import numpy as np
import cv2
import skimage.draw
from shapely import affinity
from shapely.geometry import Polygon

def normaliziraj_konturo(kontura):
    if len(kontura) <= 1:
        return np.zeros_like(kontura, dtype=float)
    kontura = np.array(kontura, dtype=float)
    kontura[:, 0] = kontura[:, 0] - np.average(kontura[:, 0])
    kontura[:, 1] = kontura[:, 1] - np.average(kontura[:, 1])

    absKontura = np.array(np.absolute(kontura))

    return kontura / absKontura.max()


def discretizeConture(kontura1, kontura2, locljivost):
    kontura1 = normaliziraj_konturo(kontura1)
    kontura2 = normaliziraj_konturo(kontura2)
    kontura1 = kontura1*locljivost/2
    kontura2 = kontura2*locljivost/2
    min = np.amin([np.amin(kontura1),np.amin(kontura2)])
    kontura1 = kontura1 - min
    kontura2 = kontura2 - min
    return np.array(kontura1, dtype=int), np.array(kontura2, dtype=int)


def primerjaj_konturi(kontura_ref, kontura_primer, locljivost=100, koraki=1):
    kontura_ref = np.array(kontura_ref)
    kontura_primer = np.array(kontura_primer)
    kontura_primer_polygon = Polygon(kontura_primer)
    grade = 0.0
    for i in range(koraki):
        angle = -(2*math.pi / koraki) * i
        rotated_polygon = affinity.rotate(kontura_primer_polygon, angle, origin="center", use_radians=True)
        rotated_conture = np.array(rotated_polygon.exterior.coords.xy).transpose()[0:-1]
        rotated_conture_dis, ref_conture_dis = discretizeConture(rotated_conture, kontura_ref, locljivost)
        rotated_mask = skimage.draw.polygon2mask([locljivost,locljivost], rotated_conture_dis)
        ref_mask = skimage.draw.polygon2mask([locljivost,locljivost], ref_conture_dis)
        intersection = np.count_nonzero(np.array(rotated_mask*ref_mask > 0))
        union = np.count_nonzero(np.array(rotated_mask+ref_mask > 0))
        if(intersection/union > grade):
            grade = intersection/union
            best_angle = -angle

    return grade, best_angle


def findLargestArea(contourRef):
    largestAreaContour = []
    largestArea = 0
    for c in contourRef:
        contour = np.array(c[:,0])
        area = cv2.contourArea(contour)
        if(largestArea < area):
            largestArea = area
            largestAreaContour = contour
    return largestAreaContour


def poisci_ujemajoce_oblike(maska, maska_ref, min_ocena=0.5, min_povrsina=100, st_korakov=100,
                            locljivost_primerjave=100):
    maska = np.array(maska, dtype="uint8")
    maska_ref = np.array(maska_ref, dtype="uint8")
    contours, _ = cv2.findContours(maska, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contourRef, _= cv2.findContours(maska_ref, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contourRef = findLargestArea(contourRef)
    contoursToReturn = []
    gradeToReturn = []
    angleToReturn = []

    for c in contours:
        contour = np.array(c[:,0])
        if cv2.contourArea(contour) < min_povrsina:
            continue
        grade, angle = primerjaj_konturi(contourRef, contour, locljivost_primerjave, st_korakov)
        if grade < 0.5:
            continue
        contoursToReturn.append(contour)
        gradeToReturn.append(grade)
        angleToReturn.append(angle)

    return contoursToReturn, gradeToReturn, angleToReturn
