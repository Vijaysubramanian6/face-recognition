import cv2

img = cv2.imread("ishan5.PNG")


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray_img.show()

cv2.imshow("IMage", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("GRAY IMage", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
