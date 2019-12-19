import unittest
from bot import remove_background
from shutil import copyfile
import os
import cv2


class NamesTestCase(unittest.TestCase):

    def test_background_removing(self):
        if (os.path.exists('test/original1.png')):
            os.remove('test/original1.png')
        copyfile('test/original.png', 'test/original1.png')
        remove_background('test/original1.png')

        original = cv2.imread('test/originalWithoutBackground.png')
        duplicate = cv2.imread("test/original1.png")
        if original.shape != duplicate.shape:
            self.assertTrue(False)

        difference = cv2.subtract(original, duplicate)
        b, g, r = cv2.split(difference)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
