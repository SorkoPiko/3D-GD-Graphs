import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def ease_in_custom(npArray):
    power = 2.056
    returnArray = []
    for x in npArray:
        if x < 0:
            x = -x
        returnArray.append(pow(x, power))
    return returnArray


def linear_easing(npArray):
    return np.divide(npArray, 1.056)


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

centers.sort(key=lambda _: _[1])

current_row: list[tuple[int, int]] = []
rows: list[list[tuple[int, int]]] = []
columns: dict[int, list[tuple[int, int]]] = {}
prev_y = centers[0][1]
threshold = 80

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
    current_row.sort(key=lambda _: _[0])
    columns[center[0]].sort(key=lambda _: _[1])

rows.append(current_row)
columns = dict(sorted(columns.items()))

row_x_y_values = []
left_column = columns[list(columns.keys())[0]]

top_row = rows[0]

for i in range(len(top_row)):
    if i > 0:
        row_x_y_values.append(top_row[i][0] - top_row[i - 1][0])

row_y_y_values = [center[1] for center in top_row]

row_y_big = min(row_y_y_values)
row_y_small = max(row_y_y_values)
row_x_big = max(row_x_y_values)
row_x_small = min(row_x_y_values)

row_y_graph = []
row_x_graph = []
row_x_graph_raw = [x for x in row_x_y_values]

row_x_raw = [x[0] for x in top_row]

for x, y in top_row:
    row_y_graph.append((y - row_y_small) / (row_y_big - row_y_small))

for x in row_x_y_values:
    row_x_graph.append((x - row_x_small) / (row_x_big - row_x_small))

y_custom = ease_in_custom(np.linspace(-1, 1, len(top_row)))
x_custom = ease_in_custom(np.linspace(-1, 1, len(row_x_y_values)))

new_list = [row_x_raw[0]]

for i in range(len(row_x_raw) - 1):
    new_list.append(row_x_raw[i] + row_x_graph_raw[i] / (row_x_graph[i] + 1))

plt.plot([i for i in range(len(top_row))], y_custom, label='Simulated Easing')
plt.plot([i for i in range(len(top_row))], row_y_graph, label='Top Row Y Pos (Render)')
plt.title('Top Row Y Position')
plt.xlabel('Dot Number')
plt.ylabel('Y Pos (Relative)')
plt.grid(True)
plt.legend()
plt.show()

plt.plot([i for i in range(len(row_x_y_values))], x_custom, label='Simulated Easing')
plt.plot([i for i in range(len(row_x_y_values))], row_x_graph, label='Top Row X Dif (Render)')
plt.title('Top Row X Difference')
plt.xlabel('Dot Number')
plt.ylabel('X Difference')
plt.grid(True)
plt.legend()
plt.show()

column_x_values = [i for i in range(len(left_column) - 1)]
column_x_x_values = [center[0] for center in left_column]
column_y_values = []

for i in range(len(left_column)):
    column_y_values.append(left_column[i][1] - rows[i][len(rows[i]) // 2][1])

print(column_y_values)

column_y_big = max(column_y_values)
column_y_small = min(column_y_values)

column_y_graph = []

for i in range(len(left_column) - 1):
    column_y_graph.append((column_y_values[i] - column_y_small) / (column_y_big - column_y_small))

print(column_y_graph)

t = np.linspace(0, 1, len(left_column) - 1)
y_custom = linear_easing(t)

plt.plot(column_x_values, y_custom, label='Simulated Easing')
plt.plot(column_x_values, column_y_graph, label='Left Column Y Dev (Render)')
plt.title('Left Column Y Deviation')
plt.xlabel('Dot Number')
plt.ylabel('Difference')
plt.grid(True)
plt.legend()
plt.show()
