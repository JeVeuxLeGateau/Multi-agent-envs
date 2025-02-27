import math
import calc as c


def detection_rate(max_rate, detection_radius, dist_from_center):
    if 1 <= dist_from_center <= detection_radius:
        return (max_rate / (dist_from_center ** 2)) * math.e ** (dist_from_center / detection_radius)
    elif 0 <= dist_from_center < 1:
        return max_rate
    else:
        return 0


def euclidean_distance_3d(x1, x2, y1, y2, z1, z2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))


class Probability_field:
    def __init__(self):
        self.field = {}

    def add_radar(self, values, max_detection_rate, position, radius):
        x = values[0]
        y = values[1]
        z = values[2]
        radar_x = position[0]
        radar_y = position[1]
        radar_z = position[2]

        for i in range(len(values[0])):
            dist_from_center = euclidean_distance_3d(radar_x, x[i], radar_y, y[i], radar_z, z[i])
            lam_d = detection_rate(max_detection_rate, radius, dist_from_center)

            if (x[i], y[i], z[i]) in self.field:
                self.field[(x[i], y[i], z[i])] = max(self.field[(x[i], y[i], z[i])], lam_d)
            else:
                self.field[(x[i], y[i], z[i])] = lam_d

        print()
