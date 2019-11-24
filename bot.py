from PIL import Image
from telebot import types
from telebot.types import Message
from flask import Flask, request

import cv2
import logging
import numpy as np
import os
import requests
import telebot


logging.basicConfig(filename="errors.log", level=logging.INFO)

TOKEN = '<TOKEN>'
STICKER_ID = 'CAADAgADBAADgqoRDwABPpw4HAMU2QI'


bot = telebot.TeleBot(TOKEN)
server = Flask(__name__)
USERS = set()


def filterOutSaltPepperNoise(edgeImg):
    # Get rid of salt & pepper noise.
    count = 0
    lastMedian = edgeImg
    median = cv2.medianBlur(edgeImg, 3)
    while not np.array_equal(lastMedian, median):
        # get those pixels that gets zeroed out
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0

        count = count + 1
        if count > 70:
            break
        lastMedian = median
        median = cv2.medianBlur(edgeImg, 3)


def findSignificantContour(edgeImg):
    contours, hierarchy = cv2.findContours(
        edgeImg,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)

    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])

    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour        


def remove_background():
    basewidth = 512
    img = Image.open('images/image.png')
    #wpercent = (basewidth / float(img.size[0]))
    #hsize = int((float(img.size[1]) * float(wpercent)))
    #img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = img.resize((basewidth, basewidth), Image.ANTIALIAS)
    img.save('images/image.png')


    src = cv2.imread('images/image.png', 1)
    blurred = cv2.GaussianBlur(src, (5, 5), 0)
    blurred_float = blurred.astype(np.float32) / 255.0
    edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
    edges = edgeDetector.detectEdges(blurred_float) * 255.0
    #cv2.imwrite('images/edge-raw.png', edges)

    edges_8u = np.asarray(edges, np.uint8)
    filterOutSaltPepperNoise(edges_8u)
    #cv2.imwrite('images/edge.png', edges_8u)
    
    contour = findSignificantContour(edges_8u)
    contourImg = np.copy(src)
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    #cv2.imwrite('images/contour.png', contourImg)
    
    mask = np.zeros_like(edges_8u)
    cv2.fillPoly(mask, [contour], 255)

    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

    # mark inital mask as "probably background"
    # and mapFg as sure foreground
    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 255] = cv2.GC_PR_BGD
    trimap[mapFg == 255] = cv2.GC_FGD

    # visualize trimap
    trimap_print = np.copy(trimap)
    trimap_print[trimap_print == cv2.GC_PR_BGD] = 128
    trimap_print[trimap_print == cv2.GC_FGD] = 255
    #cv2.imwrite('images/trimap.png', trimap_print)

    # run grabcut

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
    cv2.grabCut(src, trimap, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # create mask again
    mask2 = np.where(
        (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),
        255,
        0
    ).astype('uint8')
    #cv2.imwrite('images/mask2.png', mask2)

    contour2 = findSignificantContour(mask2)
    mask3 = np.zeros_like(mask2)
    cv2.fillPoly(mask3, [contour2], 255)

    # blended alpha cut-out
    mask3 = np.repeat(mask3[:, :, np.newaxis], 3, axis=2)
    mask4 = cv2.GaussianBlur(mask3, (3, 3), 0)
    alpha = mask4.astype(float) * 1.1  # making blend stronger
    alpha[mask3 > 0] = 255
    alpha[alpha > 255] = 255
    alpha = alpha.astype(float)

    foreground = np.copy(src).astype(float)
    foreground[mask4 == 0] = 0
    background = np.ones_like(foreground, dtype=float) * 255

    #cv2.imwrite('images/foreground.png', foreground)
    #cv2.imwrite('images/background.png', background)
    #cv2.imwrite('images/alpha.png', alpha)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha / 255.0
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    cv2.imwrite("images/foreground.png", foreground)

    src = cv2.imread("images/foreground.png", 1)
    os.remove("images/foreground.png")
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    cv2.imwrite("images/image.png", dst)


# reaction to commands


@bot.message_handler(commands=['start', 'help'])
def command_handler(message: Message):
    if 'start' in message.text:
        bot.reply_to(message, 'Start command')
        logging.info("Command @start@ triggered")
    
    if 'help' in message.text:
        bot.reply_to(message, 'Help command')
        logging.info("Command @help@ triggered")
    bot.send_message(message.chat.id, "Send me some of your shitty pics :) I'm gonna make them better!")


# reaction to text


@bot.edited_message_handler(content_types=['text'])
@bot.message_handler(content_types=['text'])
def echo_text(message: Message):
    if message.from_user.id in USERS:
        bot.send_message(message.chat.id, "Only photos dumbass")
    else:
        reply = f" Hello there, {message.from_user.first_name}"
        reply += '\nSend me some of your shitty pics :) I''m gonna make them better!\n'
        bot.send_message(message.chat.id, reply)
    USERS.add(message.from_user.id)


# reaction to photo


@bot.message_handler(content_types=['photo'])
def photo_handler(message: Message):

    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)
    logging.info(f"Image successfully downloaded (id={fileID}, file_path={file_info.file_path})")

    if not os.path.exists('images'):
        os.makedirs('images')

    with open("images/image.png", 'wb') as new_file:
        new_file.write(downloaded_file)

    bot.send_message(message.chat.id, "Did some magic here")
    remove_background()
    bot.send_document(message.chat.id, open('images/image.png', 'rb'))
    bot.send_message(message.chat.id, "Send more nudes")


@server.route('/' + TOKEN, methods=['POST'])
def getMessage():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200


@server.route("/")
def webhook():
    bot.remove_webhook()
    bot.set_webhook(url='https://your_heroku_project.com/' + TOKEN)
    return "!", 200


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
