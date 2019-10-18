from adafruit_crickit import crickit
from adafruit_seesaw.neopixel import NeoPixel
pixels = NeoPixel(crickit.seesaw, 20, 100)
pixels.fill((0,0,0))
