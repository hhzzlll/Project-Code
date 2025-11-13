import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Camera parameters
fx = 1390.85170925496
fy = 1392.93413314830
cx = 970.230107827152
cy = 601.986086684697
width = 1920
height = 1080

# Camera coordinate
z0 = 10  # [m]
L = 0.3  # body segment length [m]
lambda_val = 1

# Commented out original code (equivalent to MATLAB comments)
# xl = (-cx*z0)/fx
# xu = (width*z0-cx*z0)/fx
# theta, x0 = np.meshgrid(np.arange(0, 2*np.pi, 0.015), np.arange(xl, xu, 0.1))
# err = computeError(theta, x0, fx, z0, L, lambda_val)
# 
# # Plot
# contourLevels = np.arange(0, 35, 5)
# plt.contour(x0, np.rad2deg(theta), err, levels=contourLevels)
# plt.xlabel("x [m] (camera coordinate)")
# plt.ylabel("theta [deg]")
# plt.title(f"distance to camera (z_0)={z0}m, lambda={lambda_val}")
# plt.grid(True, alpha=0.3)

theta_range = np.linspace(0, np.pi, 100)  # 100 points from 0 to pi
phi_range = np.linspace(0, 2*np.pi, 100)  # 100 points from 0 to 2*pi

xl = (-cx*z0)/fx
xu = (width*z0-cx*z0)/fx

yl = (-cy*z0)/fy
yu = (height*z0-cy*z0)/fy

x0 = 0
y0 = -0.1

theta, phi = np.meshgrid(np.arange(0, 2*np.pi, 0.1), np.arange(0, 2*np.pi, 0.1))

lx = L * np.sin(theta) * np.cos(phi)
ly = L * np.sin(theta) * np.sin(phi)
lz = L * np.cos(theta)

def myfunc(lx, ly, lz, x0, y0, z0, method):
    fx_local = 1390.85170925496
    fy_local = 1392.93413314830

    x_apprx = fx_local * lx / (z0 + lz)
    y_apprx = fy_local * ly / (z0 + lz)

    x_actual = (fx_local * (x0 + lx) / (z0 + lz) - fx_local * x0 / z0)
    y_actual = (fy_local * (y0 + ly) / (z0 + lz) - fy_local * y0 / z0)

    if np.mean(lx) < np.mean(ly):
        actual = myDiv(x_actual, y_actual, method)
        approx = myDiv(x_apprx, y_apprx, method)
    else:
        actual = myDiv(y_actual, x_actual, method)
        approx = myDiv(y_apprx, x_apprx, method)

    err = np.abs(actual - approx)
    return err

def myDiv(y, x, method):
    if method == 'atan2':
        print('atan2')
        return np.arctan2(y, x)
    elif method == 'div':
        print('div')
        return x / y

error = myfunc(lx, ly, lz, x0, y0, z0, 'div')

plt.figure()
contourLevels = np.arange(0, 1100, 100)
contour = plt.contour(theta, phi, error, levels=contourLevels)
plt.clabel(contour, inline=True, fontsize=8)
plt.xlabel("theta")
plt.ylabel("phi")
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

# 3D scatter plot (commented equivalent)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(lx, ly, lz, c=error, cmap='viridis')
# ax.set_xlabel("lx")
# ax.set_ylabel("ly")
# ax.set_zlabel("lz")
# plt.colorbar(scatter)
# plt.show()

# Approximation error section
fx = 1390.85170925496
fy = 1392.93413314830
cx = 970.230107827152
cy = 601.986086684697
width = 1920
height = 1080

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

z0 = 0
for i in range(1):
    z0 = z0 + 10
    l = 0.3
    
    d, theta = np.meshgrid(np.arange(-width/2, width/2, 0.1), np.arange(0, 2*np.pi, 0.1))
    
    xv = l * np.cos(theta)
    zv = l * np.sin(theta)
    x = (fx * xv - d * zv) / (z0 + zv)
    x_approx = (fx * xv) / (z0 + zv)
    err = np.abs(x_approx - x)

    contourLevels = np.arange(0, 22, 2)
    contour = plt.contour(d, zv, err, levels=contourLevels, linewidths=1)
    plt.xlabel("horizontal offset from image center [px]")
    plt.ylabel("z_a [m]")
    plt.title(f"Approximation error with z_0 = {z0:.2f} m, arm length = {l:.2f} m")
    plt.grid(True, alpha=0.3)

plt.show()
