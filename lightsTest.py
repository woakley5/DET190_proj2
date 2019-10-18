from adafruit_crickit import crickit
from adafruit_seesaw.neopixel import NeoPixel
import time
numLights = 100
pixels = NeoPixel(crickit.seesaw, 20, numLights)
print("Testing " + str(numLights) + " lights.")
r = range(numLights)
for i in r[::3]:
    print(i)
    pixels[i] = (0,255,0)
    time.sleep(0.25)
    pixels[i] = (0,0,0)
    time.sleep(0.25)

pixels.fill((0,0,0))
