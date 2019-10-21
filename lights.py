from adafruit_crickit import crickit
from adafruit_seesaw.neopixel import NeoPixel
import time

class Lights():
    def __init__(self, num_pixels):
        self.pixels = NeoPixel(crickit.seesaw, 20, num_pixels)
        self.pixels.fill((0,0,0))
        self.num_pixels = num_pixels
        self.pixels.fill((0,0,0))

    def updateLEDs(self, newColor, delay):
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
        self.pixels.fill((r, g, b))
        time.sleep(delay)

    def fillEach(self, color, delay):
        for x in range(self.num_pixels):
            self.pixels[x] = color
            time.sleep(delay)
