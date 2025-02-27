import numpy as np
import matplotlib.pyplot as plt


class graph_scene:
    def __init__(self):
        # Create figure and 3D axis
        self.fig = plt.figure(figsize=(20, 20))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Set Size
        self.ax.set_xlim([0, 1000])
        self.ax.set_ylim([0, 1000])
        self.ax.set_zlim([0, 1000])

        # Labels
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')
        self.ax.set_title('3D Surface Plot')

        self.shapes = []

    def create_sphere(self, center, radius=10, color="black"):
        x_center = center[0]
        y_center = center[1]
        z_center = center[2]

        # Generate grid points
        x_vals = np.arange(x_center - radius, x_center + radius + 1)
        y_vals = np.arange(y_center - radius, y_center + radius + 1)
        z_vals = np.arange(z_center - radius, z_center + radius + 1)

        # Create empty arrays for sphere points
        x, y, z = [], [], []

        # Iterate through grid and apply sphere equation
        for i in x_vals:
            for j in y_vals:
                for k in z_vals:
                    if (i - x_center) ** 2 + (j - y_center) ** 2 + (k - z_center) ** 2 <= radius ** 2:
                        x.append(i)
                        y.append(j)
                        z.append(k)

        self.shapes.append(((x, y, z), color))

    def create_rectangle(self, blc, length, width, height, color="black"):
        x_center = blc[0]
        y_center = blc[1]
        z_center = blc[2]

        # Generate grid points
        x_vals = np.arange(x_center, x_center + length + 1)
        y_vals = np.arange(y_center, y_center + width + 1)
        z_vals = np.arange(z_center, z_center + height + 1)

        # Create empty arrays for sphere points
        x, y, z = [], [], []

        # Iterate through grid and apply sphere equation
        for i in x_vals:
            for j in y_vals:
                for k in z_vals:
                    x.append(i)
                    y.append(j)
                    z.append(k)

        self.shapes.append(((x, y, z), color))

    def create_line(self, A, B, color="black"):
        A = np.array(A)
        B = np.array(B)

        # Compute distance between A and B
        dist = np.linalg.norm(B - A)

        # Compute number of steps
        num_points = int(dist) + 1  # Ensure at least two points

        # Generate points using linear interpolation
        x_vals = np.linspace(A[0], B[0], num_points)
        y_vals = np.linspace(A[1], B[1], num_points)
        z_vals = np.linspace(A[2], B[2], num_points)

        self.shapes.append(((x_vals, y_vals, z_vals), color))

    def render_shapes(self):
        for shape in self.shapes:
            shape_positions = shape[0]
            shape_color = shape[1]

            x = shape_positions[0]
            y = shape_positions[1]
            z = shape_positions[2]

            self.ax.scatter(x, y, z, color=shape_color)

        plt.show()



