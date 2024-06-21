from PIL import Image, ImageDraw
import numpy as np

X_DISTANCE = 10
ROW_Y_DISTANCE = 1

def ease_in_custom(npArray):
    power = 2.058
    returnArray = []
    for x in npArray:
        if x < 0:
            returnArray.append(-pow(-x, power))
        else:
            returnArray.append(pow(x, power))
    return returnArray

def ease_in_pos(npArray):
    power = 2.058
    returnArray = []
    for x in npArray:
        if x < 0:
            x = -x
        returnArray.append(pow(x, power))
    return returnArray


# Create a new image with the size of 1920x1080 pixels (1080p)
image = Image.new('RGB', (1920, 1080), 'white')

t = ease_in_custom(np.linspace(-X_DISTANCE, X_DISTANCE, 160))
e = ease_in_pos(np.linspace(-ROW_Y_DISTANCE, ROW_Y_DISTANCE, 160))
c = ease_in_custom(np.linspace(-X_DISTANCE, X_DISTANCE, 90))

# Create a draw object from the image
draw = ImageDraw.Draw(image)

# Calculate the distance between the dots
x_distance = 1920 // 160
y_distance = 1080 // 90

# Draw the dots on the image
for j in range(90):
    for i in range(160):
        x = i * x_distance + 1920 / 160 / 2
        y = j * y_distance + c[j] * e[i] + 1080 / 90 / 2
        draw.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill='black')

# Save the image to a file
image.save('dot_grid.png')
