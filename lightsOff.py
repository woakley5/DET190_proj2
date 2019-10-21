from adafruit_crickit import crickit
from adafruit_seesaw.neopixel import NeoPixel
import time

num_pixels = 75
pixels = NeoPixel(crickit.seesaw, 20, num_pixels)
pixels.fill((0,0,0))

def updateLEDs(newColor, delay):
    r = 0
    g = 0
    b = 0
    while(r < newColor[0] or g < newColor[1] or b < newColor[2]):
        if(r < newColor[0]):
            r += 1
        if(g < newColor[1]):
            g += 1
        if(b < newColor[2]):
            b += 1
        pixels.fill((r, g, b))
        time.sleep(delay)

def fillEach(color, delay):
    for x in range(num_pixels):
        pixels[x] = color
        time.sleep(delay)

updateLEDs((50,0,50), 0.1)
fillEach((0,50,0), 0.1)
