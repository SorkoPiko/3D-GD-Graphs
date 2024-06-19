import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def ease_in_custom(t):
    return np.power(t, 1.85)


input_folder = 'input'

image = cv2.imread(f'{input_folder}/{os.listdir(input_folder)[0]}')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

centers = []

for contour in contours:
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centers.append((cX, cY))

centers.sort(key=lambda x: x[1])

rows = []
columns = {}
current_row = []
prev_y = centers[0][1]
threshold = 10

for center in centers:
    if abs(center[1] - prev_y) > threshold:
        rows.append(current_row)
        current_row = []
    if center[0] not in columns:
        columns[center[0]] = [center]
    else:
        columns[center[0]].append(center)
    current_row.append(center)
    prev_y = center[1]
    current_row.sort(key=lambda x: x[0])

rows.append(current_row)

print(columns)

new_image = np.zeros_like(image)

x_dist = []

e = 0
for row in rows:
    if e == 0:
        colour = (0, 255, 0)
    else:
        colour = (0, 0, 255)
    for i in range(len(row)):
        cv2.circle(new_image, row[i], 5, colour, -1)

        if i < len(row) - 1:
            cv2.line(new_image, row[i], row[i + 1], colour, 2)
            if e == 0:
                x_dist.append(row[i + 1][0] - row[i][0])
    e += 1

cv2.imwrite('output.png', new_image)

top_row = rows[0]

x_values = [i + round(len(top_row) / 2) for i in range(round(len(top_row) / 2))]
y_y_values = [center[1] for center in top_row]
x_y_values = x_dist

y_big = min(y_y_values)
y_small = max(y_y_values)
x_big = max(x_y_values)
x_small = min(x_y_values)

y_graph = []
x_graph = []

i = 0
for x, y in top_row:
    if i >= len(top_row) / 2:
        y_graph.append((y - y_small) / (y_big - y_small))
        x_graph.append((x_dist[i - 1] - x_small) / (x_big - x_small))
    i += 1

t = np.linspace(0, 1, round(len(top_row) / 2))
y_custom = ease_in_custom(t)

plt.plot(x_values, y_custom, label='Simulated Easing')
plt.plot(x_values, y_graph, label='Top Row Y Dif (Render)')
plt.plot(x_values, x_graph, label='Top Row X Dif (Render)')
plt.title('Top Row X/Y Difference')
plt.xlabel('Dot Number')
plt.ylabel('Difference')
plt.grid(True)
plt.legend()
plt.show()