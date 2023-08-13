import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import matplotlib.font_manager as fm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image

class CapturePlate_CharacterSegmentation():
  def normalize_image(self, snapshot):
    maxvalue=float(snapshot.max())
    minvalue=float(snapshot.min())
    for i in range(snapshot.shape[0]):
       for j in range(snapshot.shape[1]):
          snapshot[i, j]=(255 / (maxvalue - minvalue) * snapshot[i, j] - (255 * minvalue) / (maxvalue - minvalue))
    return snapshot
  def threshold_image(self,normalize):
    maxvalue=float(normalize.max())
    minvalue=float(normalize.min())
    ret1=maxvalue-(maxvalue-minvalue)/2
    ret1,thresh1=cv.threshold(normalize,ret1,255,cv.THRESH_BINARY)
    return thresh1
  def get_contour_bbox(self, contour0):
        x, y = [], []
        for p in contour0:
            x.append(p[0][0])
            y.append(p[0][1])
        return [min(x), min(y), max(x), max(y)]
  def largest_block_bbox(self,imge,after0):
      contours1, hierarchy = cv.findContours(imge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
      block0 = []
      for c in contours1:
          r = self.get_contour_bbox(c)
          a = (r[2] - r[0]) * (r[3] - r[1])  # area
          l = (r[2] - r[0]) * (r[3] - r[1])  # aspect ratio
          block0.append([r, a, l])
      block0 = sorted(block0, key=lambda b: b[1])[-3:]
      maxweight, maxindex = 0, -1
      for i in range(len(block0)):
          b = after0[block0[i][0][1]:block0[i][0][3], block0[i][0][0]:block0[i][0][2]]
          hsv = cv.cvtColor(b, cv.COLOR_BGR2HSV)
          lower = np.array([100, 50, 50])
          upper = np.array([140, 255, 255])
          mask = cv.inRange(hsv, lower, upper)
          w0 = 0
          for m in mask:
              w0 += m / 255
          w1 = 0
          for n in w0:
              w1 += n
          if w1 > maxweight:
              maxindex = i
              maxweight = w1
      return block0[maxindex][0]
  def plate_processing(self,snapshot):
    plt.figure("Function: plate_processing")
    plt.suptitle('Function: plate_processing')
    plt.subplot(7, 3, 1)
    plt.title('1_Original RGB', fontsize=9)
    plt.imshow(cv.cvtColor(snapshot, cv.COLOR_BGR2RGB))
    plt.axis('off')

    snapshot=cv.resize(snapshot,(400,int(400 * snapshot.shape[0] / snapshot.shape[1])))
    plt.subplot(7, 3, 2)
    plt.title('2_Resized Image BGR', fontsize=9)
    plt.imshow(snapshot)
    plt.axis('off')

    gray_image=cv.cvtColor(snapshot,cv.COLOR_BGR2GRAY)
    plt.subplot(7, 3, 3)
    plt.title('3_Grayscale Image', fontsize=9)
    plt.imshow(gray_image)
    plt.axis('off')

    normalized_image=self.normalize_image(gray_image)
    plt.subplot(7, 3, 7)
    plt.title('4_Normalized Image', fontsize=9)
    plt.imshow(normalized_image)
    plt.axis('off')

    r = 16
    h = w = r * 2 + 1
    kernel = np.zeros((h, w), np.uint8)
    cv.circle(kernel, (r, r), r, 1, -1)
    opening_image=cv.morphologyEx(normalized_image,cv.MORPH_OPEN,kernel)
    plt.subplot(7, 3, 8)
    plt.title('5_Opening1 Image', fontsize=9)
    plt.imshow(opening_image)
    plt.axis('off')

    strtimage=cv.absdiff(normalized_image,opening_image)
    plt.subplot(7, 3, 9)
    plt.title('6_Absdiff Image', fontsize=9)
    plt.imshow(strtimage)
    plt.axis('off')

    binary_image=self.threshold_image(strtimage)
    plt.subplot(7, 3, 13)
    plt.title('7_Binary Image', fontsize=9)
    plt.imshow(binary_image)
    plt.axis('off')

    canny=cv.Canny(binary_image,binary_image.shape[0],binary_image.shape[1])
    kernel=np.ones((5,24),np.uint8)
    closing_image=cv.morphologyEx(canny,cv.MORPH_CLOSE,kernel)
    plt.subplot(7, 3, 14)
    plt.title('8_Closing Image', fontsize=9)
    plt.imshow(closing_image)
    plt.axis('off')

    opening_image=cv.morphologyEx(closing_image,cv.MORPH_OPEN,kernel)
    plt.subplot(7, 3, 15)
    plt.title('9_Opening2 Image', fontsize=9)
    plt.imshow(opening_image)
    plt.axis('off')

    kernel=np.ones((11,6),np.uint8)
    opening_image=cv.morphologyEx(opening_image,cv.MORPH_OPEN,kernel)
    plt.subplot(7, 3, 19)
    plt.title('10_Opening3 Image', fontsize=9)
    plt.imshow(opening_image)
    plt.axis('off')

    rect0=self.largest_block_bbox(opening_image,snapshot)
    x1 = max(0, int(rect0[0]) - 5)
    x2 = min(snapshot.shape[1] - 1, int(rect0[2]) + 0)
    y1 = max(0, int(rect0[1]) - 0)
    y2 = min(snapshot.shape[0] - 1, int(rect0[3]) + 5)
    image2_with_box = snapshot.copy()
    x1, y1, x2, y2 = int(rect0[0]), int(rect0[1]), int(rect0[2]), int(rect0[3])
    plt.subplot(7, 3, 20)
    plt.title('11_Plate framing', fontsize=9)
    cv.rectangle(image2_with_box, (x1-3, y1-10), (x2+3, y2+10), (0, 255, 0), 2)
    plt.imshow(image2_with_box)
    plt.axis('off')
    plt.savefig("D:/lunwen/CarIdRecognition/imgs/plate_processing.jpg", bbox_inches='tight')
    plt.show()

    return rect0, snapshot
  def grabcut_segmentation(self, after0, rect0):
        # Adjust the left and right trimming range
        rect0[0] -= 10
        rect0[2] += 10
        # Adjust the top and bottom trimming range
        rect0[1] -= 20
        rect0[3] += 50
        #Convert to width and height
        rect0[2] = rect0[2] - rect0[0]
        rect0[3] = rect0[3] - rect0[1]
        rect_copy = tuple(rect0.copy())
        mask0 = np.zeros(after0.shape[:2], np.uint8)
        #Create a background model with a size of 13x5, one row, and single-channel floating-point type.
        bgdModel = np.zeros((1, 65), np.float64)
        #Create a foreground model
        fgdModel = np.zeros((1, 65), np.float64)
        cv.grabCut(after0, mask0, rect_copy, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
        mask1 = np.where((mask0 == 2) | (mask0 == 0), 0, 1).astype('uint8')
        image_show = after0 * mask1[:, :, np.newaxis]
        return image_show
  def segment_and_save_characters(self,path):
        image = cv.imread(path)
        plt.figure("character segmentation1")
        plt.suptitle("character segmentation1")
        plt.subplot(10, 1, 1)
        plt.title('1_Plate', fontsize=9)
        plt.imshow(image, cmap='gray')


        blurerd = cv.GaussianBlur(image, (3, 3), 0, 0, cv.BORDER_DEFAULT)
        plt.subplot(10, 1, 3)
        plt.title('2_GaussianBlur', fontsize=9)
        plt.imshow(blurerd, cmap='gray')


        image_gray = cv.cvtColor(blurerd, cv.COLOR_BGR2GRAY)
        plt.subplot(10, 1, 5)
        plt.title('3_BGR2GRAY', fontsize=9)
        plt.imshow(image_gray, cmap='gray')

        # Binary image
        maxvalue = float(image_gray.max())
        minvalue = float(image_gray.min())
        ret = maxvalue - (maxvalue - minvalue) / 2
        ret, thresh = cv.threshold(image_gray, ret, 255, cv.THRESH_BINARY)
        binary_image = thresh
        plt.subplot(10, 1, 7)
        plt.title('4_Binary', fontsize=9)
        plt.imshow(binary_image)

        # Morphological operations
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        image = cv.dilate(binary_image, kernel)
        plt.subplot(10, 1, 9)
        plt.title('5_Dilate', fontsize=9)
        plt.imshow(image)
        plt.savefig("D:/lunwen/CarIdRecognition/imgs/segment_characters1.jpg", bbox_inches='tight')
        plt.show()
        # Find contours
        contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        text = []
        chara_images = []
        for item in contours:
            chara = []
            rect = cv.boundingRect(item)
            x = rect[0]
            y = rect[1]
            weight = rect[2]
            height = rect[3]
            chara.append(x)
            chara.append(y)
            chara.append(weight)
            chara.append(height)
            text.append(chara)
        text = sorted(text, key=lambda s: s[0], reverse=False)
        i = 0
        plt.figure("character segmentation2")
        plt.suptitle("character segmentation2")
        for word in text:
            # Filter contours based on the bounding rectangle of each contour
            if (word[3] > (word[2] * 1)) and (word[3] < (word[2] * 3)) and (word[2] > 10):
                i = i + 1
                splite_image = image_gray[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
                #splite_imagee = cv.cvtColor(splite_image, cv.COLOR_BGR2GRAY)
                splite_image = cv.resize(splite_image, (32, 40))
                chara_images.append(splite_image)
        save_folder = 'D:/lunwen/CarIdRecognition/imgs/characters'
        # Clear folder
        if os.path.exists(save_folder):
            for file_name in os.listdir(save_folder):
                file_path = os.path.join(save_folder, file_name)
                os.remove(file_path)
        else:
            os.makedirs(save_folder, exist_ok=True)
        for i, j in enumerate(chara_images):
            plt.subplot(1, len(chara_images), i + 1)
            plt.imshow(chara_images[i], cmap='gray')
            # Save images to folder
            image_path = os.path.join(save_folder, f'char_{i}.jpg')
            _, chara_images[i] = cv.threshold(chara_images[i], 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
            chara_images[i] = cv.bitwise_not(chara_images[i])
            cv.imwrite(image_path, chara_images[i])
        plt.savefig("D:/lunwen/CarIdRecognition/imgs/segment_characters2.jpg", bbox_inches='tight')
        plt.show()
width0 = 32
height0 = 40
image_size = 1280
classes0 = 41  # The number of image classes, i.e., the images should belong to one of 41 categories. In the end, the probability for each category should be obtained.

# The set of license plate numbers to be recognized consists of 41 classes of outputs. These classes include digits 0-9, English letters A-Z (excluding I and O), and some Chinese characters.
characters0 = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Training dataset ID 0-9
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',  # Training dataset ID 10-21  (excluding I)
              'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  # Training dataset ID 22-33 (excluding O)
              '京', '闽', '粤', '苏', '沪', '浙','津'  # Training dataset ID 34-40
               )

def license_plate_recognition():
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[None, image_size])
    y = tf.placeholder(tf.float32, shape=[None, classes0])
    x_images = tf.reshape(x, [-1, width0, height0, 1])

    W_con0 = tf.Variable(tf.random_normal([8, 8, 1, 16], stddev=0.1), name='W_con1')
    b_con0 = tf.Variable(tf.constant(0.1, shape=[16]), name='b_con1')
    jj_con0 = tf.nn.conv2d(x_images, W_con0, strides=[1, 1, 1, 1], padding="SAME")
    jh_con0 = tf.nn.relu(jj_con0 + b_con0)
    ch_con0 = tf.nn.max_pool(jh_con0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    W_con1 = tf.Variable(tf.random_normal([5, 5, 16, 32], stddev=0.1), name='W_con1')
    b_con1 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_con2')
    jj_con1 = tf.nn.conv2d(ch_con0, W_con1, strides=[1, 1, 1, 1], padding="SAME")
    jh_con1 = tf.nn.relu(jj_con1 + b_con1)
    ch_con1 = tf.nn.max_pool(jh_con1, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding="SAME")

    h_fc0_flat = tf.reshape(ch_con1, [-1, 16 * 20 * 32])
    W_fc0 = tf.Variable(tf.random_normal([16 * 20 * 32, 512], stddev=0.1), name="W_fc1")  
    b_fc0 = tf.Variable(tf.constant(0.1, shape=[512]), name="b_fc1")  
    h_fc0 = tf.nn.relu(tf.matmul(h_fc0_flat, W_fc0) + b_fc0)

    rate = tf.placeholder(tf.float32)
    h_fc0_drop = tf.nn.dropout(h_fc0, rate)

    W_fc1 = tf.Variable(tf.random_normal([512, classes0], stddev=0.1), name="W_fc2") 
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[classes0]), name="b_fc2") 
    y_con = tf.matmul(h_fc0_drop, W_fc1) + b_fc1

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # Load the saved model
        saver = tf.train.Saver()
        saver.restore(sess, "./save_model/letter_digits_model.ckpt")
        licence_num = ""  # Used to concatenate the license plate numbers
        files = os.listdir("D:/lunwen/CarIdRecognition/imgs/characters")  # Path to the folder containing test images
        num_png = len(files)
        first_recognition = True
        matched_characters_with_prob = []
        for i in range(num_png):
            path = "D:/lunwen/CarIdRecognition/imgs/characters/char_%d.jpg" % i  # Path to individual test image
            img = Image.open(path)

            width = img.size[0]
            height = img.size[1]

            img_data = [[0] * image_size for i in range(1)]

            for h in range(height):
                for w in range(width):
                    if img.getpixel((w, h)) < 190:
                        img_data[0][w + h * width] = 1
                    else:
                        img_data[0][w + h * width] = 0

            result = sess.run(y_con, feed_dict={x: np.array(img_data), rate: 1.0})

            max_prob = 0
            max_index = 0
            for j in range(classes0):
                if result[0][j] > max_prob:
                    max_prob = result[0][j]
                    max_index = j

            # Save matching characters and probabilities to the list
            matched_characters_with_prob.append((characters0[max_index], max_prob))

            if first_recognition:  # First identification for special treatment
                if max_index < 34:
                    licence_num = "*"
                else:
                    licence_num += characters0[max_index]
                first_recognition = False
            else:
                licence_num += characters0[max_index]

        # Check the length of licence_num and append "?" until it reaches 7 characters
        while len(licence_num) < 7:
            licence_num += "?"

        return licence_num, matched_characters_with_prob


def process_and_recognize_license_plate(image_path):
    # Preprocess image
    snapshot = cv.imread(image_path)
    license = CapturePlate_CharacterSegmentation()
    rect, image1 = license.plate_processing(snapshot)
    image1 = cv.cvtColor(image1, cv.COLOR_RGB2BGR)

    x1 = max(0, int(rect[0]) - 5)
    x2 = min(image1.shape[1] - 1, int(rect[2]) + 0)
    y1 = max(0, int(rect[1]) - 0)
    y2 = min(image1.shape[0] - 1, int(rect[3]) + 5)
    x1, y1, x2, y2 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    # background removing
    plate1 = license.grabcut_segmentation(image1, rect)
    plt.figure("Plate Segmentation")  # Create a new figure
    plt.suptitle('Plate Segmentation')

    plt.subplot(1, 2, 1)
    plt.title('1_Grabcut_segmentation', fontsize=9)
    plt.imshow(plate1)
    #print('background removal:', x1, y1, x2, y2)
    #Cut the license plate out of the background
    plate1 = plate1[y1-10:y2+10, x1-3:x2+3]
    height, width = plate1.shape[:2]
    plate1 = cv.resize(plate1, (2 * width, 2 * height), interpolation=cv.INTER_CUBIC)
    plt.subplot(1, 2, 2)
    plt.title('2_Split license plate', fontsize=9)
    plt.imshow(plate1)
    plt.savefig("D:/lunwen/CarIdRecognition/imgs/segment_plate.jpg", bbox_inches='tight')
    plt.show()

    # Specify the save directory
    save_directory = r'D:/lunwen/CarIdRecognition/imgs/plate'
    # Save the image to the specified directory
    save_path = os.path.join(save_directory, 'plate.jpg')
    cv.imwrite(save_path, plate1)


    path = "D:/lunwen/CarIdRecognition/imgs/plate/plate.jpg"
    #sub_plate = sub_plate0()
    #sub_plate.segment_and_save_characters(path)  # 把图片的路径给它
    license.segment_and_save_characters(path)

    #from loadCNN0714 import license_plate_recognition
    licence_num,matched_characters_with_prob = license_plate_recognition()  # loadCNN.py
    # 打印车牌号
    print("License Plate Number:", licence_num)

    # 打印每张图片的匹配结果
    for i, (character, prob) in enumerate(matched_characters_with_prob):
        print(f"Image {i}: Character = {character}, Probability = {prob:.4f}")
    if len(licence_num)==7:
        # List of license plates to recognize
        license_plates = ["粤AFQ787", "B111111", "C111111", "D111111"]
        num_matching_characters = 0
        for i, char in enumerate(licence_num):
            for plate in license_plates:
                if char == plate[i]:
                    num_matching_characters += 1
                    break
        if num_matching_characters >= 4:
            print("Open the door")  # Open the door
        else:
            print("Not on the list")  # Not on the list
    else:
        print("License plate is not 7 characters long")
        num_matching_characters=0
    return num_matching_characters
if __name__ == '__main__':
    #image_path = 'D:/lunwen/CarIdRecognition/test/test/2.jpg'
    image_path = 'D:/lunwen/CarIdRecognition/imgs/car/car/6.jpg'
    #process_and_recognize_license_plate(image_path)
    print("Num_matching:",process_and_recognize_license_plate(image_path))



#Reference:
#1：MZYYZT. (2022, December 07). [Digital Image Processing - License Plate Recognition]. CSDN. URL: [https://blog.csdn.net/MZYYZT/article/details/128212029]
#2：caishuxueqiandemoxiaotian. (2021, February 26). [ Python-OpenCV Practical Project: License Plate Recognition (Part 1) - Accurate License Plate Localization]. CSDN. URL: [https://blog.csdn.net/qq_45804132/article/details/114150432]
#3：SOBE_rrr. (2022, September 29). [Implementing license plate recognition using CNN (Convolutional Neural Network) in TensorFlow. Complete Python project code attached]. CSDN. URL: [https://blog.csdn.net/SOBE_rrr/article/details/119924079?app_version=5.15.5&code=app_1562916241&csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22119924079%22%2C%22source%22%3A%22SOBE_rrr%22%7D&uLinkId=usr1mkqgl919blen&utm_source=app]
#4：pangdahaipyh. (2020, May 26). [[OpenCV Practical] Two Methods for Character Segmentation in License Plate Recognition (OCR) Implemented in Python (Summary)]. CSDN. URL: [https://blog.csdn.net/qq_40784418/article/details/106262771]
