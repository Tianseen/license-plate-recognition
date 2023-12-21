from carDetTools import licCut

img_path = 'src/images/015-90_260-228&437_444&507-444&507_238&502_228&439_435&437-0_0_3_24_28_26_28_32-111-64.jpg'

lic, label = licCut(img_path)

print(label)

#打印图片
import cv2

img = cv2.imread(lic)


