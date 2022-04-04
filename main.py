import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


def import_image(filename):
    return cv2.imread(filename)


def convert_image_to_rgb(img):
    clone = img.copy()
    return cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)


def blur_image(img, kernel_size):
    clone = img.copy()
    return cv2.blur(clone, kernel_size)


def convert_image_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny_edges(img, threshold_1, threshold_2):
    return cv2.Canny(image=img, threshold1=threshold_1, threshold2=threshold_2)


def find_contours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE):
    contours, hierarchy = cv2.findContours(image=img, mode=mode, method=method)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours


def draw_contours_on_img(img, contours, color=(255, 0, 0), thickness=2, contour_idx=-1):
    clone = img.copy()
    return cv2.drawContours(clone, contours, contourIdx=contour_idx, color=color, thickness=thickness)


def convert_image_to_black_and_white(img, threshold, maximum_bgr, threshold_type=cv2.THRESH_BINARY_INV):
    clone = img.copy()
    _, thresh = cv2.threshold(clone, threshold, maximum_bgr, threshold_type)

    return thresh


def dilate_img(img, kernel, iterations=1):
    clone = img.copy()
    return cv2.dilate(clone, kernel, iterations=iterations)


def erode_img(img, kernel, iterations=1):
    clone = img.copy()
    return cv2.erode(clone, kernel, iterations=iterations)


def apply_morphology_on_img(img, kernel, morphology):
    clone = img.copy()
    return cv2.morphologyEx(clone, morphology, kernel)


def prepare_plot_size(images):
    images_length = len(images)

    width = math.ceil(images_length ** .5)

    if (width ** 2 - images_length) > width:
        height = width - 1
    else:
        height = width

    return width, height


def save_plot(width, height, images, output="process"):
    for i in range(len(images)):
        plt.subplot(height, width, i + 1)
        plt.imshow(images[i], 'gray')
        plt.xticks([]), plt.yticks([])
    plt.savefig(output)


def main():
    img = import_image("FEI01.jpg")
    img = convert_image_to_rgb(img)

    kernel_size = 6
    kernel_mtx = (kernel_size, kernel_size)

    img_blur = blur_image(img, kernel_mtx)
    img_gray = convert_image_to_gray(img_blur)

    max_bgr = img_gray.max()
    threshold = max_bgr/2*1.55
    print(f"Largest BGR value: {max_bgr}")
    print(f"Threshold: {threshold}")

    img_bw = convert_image_to_black_and_white(img_gray, threshold, max_bgr)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    img_dilate = dilate_img(img_bw, kernel)
    img_close = apply_morphology_on_img(img_dilate, kernel, cv2.MORPH_CLOSE)

    edges_on_last_morphology = canny_edges(img_close, threshold, threshold)
    contours = find_contours(edges_on_last_morphology)
    img_with_contours = draw_contours_on_img(img, contours)
    rgb_img_with_contours = convert_image_to_rgb(img_with_contours)
    
    images = [
        img,
        img_blur,
        img_gray,
        img_bw,
        img_dilate,
        img_close,
        edges_on_last_morphology,
        img_with_contours
    ]
    
    width, height = prepare_plot_size(images)
    save_plot(width, height, images)

    cv2.imwrite("final.png", rgb_img_with_contours)


if __name__ == "__main__":
    main()
