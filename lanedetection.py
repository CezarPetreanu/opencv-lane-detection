import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])  # bottom of the image
    y2 = int(y1 * 5 / 8)  # slightly lower than the middle
    # print("slope", slope)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    if x1 < 0:
        x1 = 0
    if x1 > image.shape[1]:
        x1 = image.shape[1]
    if x2 < 0:
        x2 = 0
    if x2 > image.shape[1]:
        x2 = image.shape[1]

    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            # print(fit)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:  # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # add more weight to longer lines
    if left_fit != []:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_points(image, left_fit_average)
    else:
        left_line = None

    if right_fit != []:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(image, right_fit_average)
    else:
        right_line = None

    averaged_lines = [left_line, right_line]
    # print(averaged_lines)
    return averaged_lines


def canny(img):
    # CONRAST
    contrast = img
    cv2.normalize(contrast, contrast, -50, 300, cv2.NORM_MINMAX)
    #

    gray = cv2.cvtColor(contrast, cv2.COLOR_RGB2GRAY)
    cv2.imshow("gray", gray)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    cv2.imshow("blur", blur)
    canny = cv2.Canny(blur, 100, 150)
    cv2.imshow("canny", canny)
    return canny


def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            if line is not None:
                # print(line)
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 10)
    return line_image


def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)

    # ce ne intereseaza
    rof = np.array(
        [
            [
                (
                    (0, height),
                    (0, height * 0.87),
                    (width * 0.43, height // 2),
                    (width * 0.57, height // 2),
                    (width, height * 0.87),
                    (width, height),
                )
            ]
        ],
        np.int32,
    )
    cv2.fillPoly(mask, rof, 255)

    # ce NU ne intereseaza (semne din mijlocul drumului)
    offset = 200
    rof = np.array(
        [
            [
                (
                    (0 + offset, height),
                    (0 + offset, height - height // 7),
                    (width * 7 // 16 + offset, height // 2),
                    (width * 9 // 16 - offset, height // 2),
                    (width - offset, height - height // 7),
                    (width - offset, height),
                )
            ]
        ],
        np.int32,
    )
    # cv2.fillPoly(mask, rof, 0)

    cv2.imshow("mask", mask)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


def main():
    # image = cv2.imread('test_image.jpg')
    image = cv2.imread("frame0.jpg")

    img = np.copy(image)
    cv2.imshow("frame", img)
    lines_canny = canny(img)
    cv2.imshow("canny", lines_canny)

    # print(img.shape)
    """
    plt.figure()
    plt.imshow(img)
    plt.show()
    """

    cropped_canny = region_of_interest(lines_canny)
    cv2.imshow("ROI-canny", cropped_canny)
    lines = cv2.HoughLinesP(
        cropped_canny, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5
    )
    averaged_lines = average_slope_intercept(image, lines)
    line_image = display_lines(img, averaged_lines)
    cv2.imshow("lines", line_image)
    combo_image = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    cv2.imshow("result", combo_image)
    cv2.waitKey(0)


def seeFrames(video):
    cap = cv2.VideoCapture(video)
    success = True
    count = 0
    while success:
        success, image = cap.read()
        cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        if cv2.waitKey(10) == 27:  # exit if Escape is hit
            break
        count += 1


def lane_in_video(video):
    cap = cv2.VideoCapture(video)

    prev_lines = None

    while cap.isOpened():
        _, fr = cap.read()
        frame = cv2.resize(fr, (1280, 720))

        canny_image = canny(frame)
        cropped_canny = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(
            cropped_canny,
            2,
            np.pi / 180,
            100,
            np.array([]),
            minLineLength=40,
            maxLineGap=5,
        )

        averaged_lines = average_slope_intercept(frame, lines)

        if (
            averaged_lines is None
            or averaged_lines[0] is None
            or averaged_lines[1] is None
        ):
            averaged_lines = prev_lines

        print(averaged_lines)

        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("result", combo_image)
        cv2.imshow("cropped", cropped_canny)

        prev_lines = averaged_lines

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # main()
    lane_in_video(0) #for webcam
    #lane_in_video("ttest.mp4") # for video test
