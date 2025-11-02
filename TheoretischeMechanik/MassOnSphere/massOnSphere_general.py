import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

R = 1.0
m = 1.0
g = 9.81

# initial conditions
theta0 = np.pi/8
theta_dot0 = 0.0
phi0 = 0.0
phi_dot0 = 1

p1 = m * R**2 * (np.sin(theta0)**2) * phi_dot0

def dSdt(t, S0):
    theta, thdot, phi = S0
    s = np.sin(theta)
    if abs(s) < 1e-8: s = 1e-8
    phi_dot = p1 / (m * R**2 * s**2)
    theta_ddot = np.sin(theta)*np.cos(theta) * phi_dot**2 + (g/R)*np.sin(theta)
    return [thdot, theta_ddot, phi_dot]

t_span = (0.0, 6.0)
t_eval = np.linspace(t_span[0], t_span[1], 800)
S0 = [theta0, theta_dot0, phi0]
sol = solve_ivp(dSdt, t_span, S0, t_eval=t_eval, rtol=1e-9, atol=1e-12)

theta = sol.y[0]
phi = sol.y[2]
time = sol.t

x = R*np.sin(theta)*np.cos(phi)
y = R*np.sin(theta)*np.sin(phi)
z = R*np.cos(theta)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# plot sphere
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
X = R * np.outer(np.cos(u), np.sin(v))
Y = R * np.outer(np.sin(u), np.sin(v))
Z = R * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(X, Y, Z, color='c', alpha=0.1)  

# Trajecory
ax.plot(x, y, z, 'r-', label='Trajectory')

ax.scatter(x[-1], y[-1], z[-1], color='b', s=50, label='Point')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1,1,1]) 
ax.legend()

plt.show()
exit(0)

plt.figure()
plt.plot(time, theta)
plt.xlabel('t (s)'); plt.ylabel(r'$\theta(t)$'); plt.title('Polar angle')

plt.figure()
plt.plot(time, phi)
plt.xlabel('t (s)'); plt.ylabel(r'$\varphi(t)$'); plt.title('Azimuthal angle')

plt.figure()
plt.plot(phi, theta)
plt.xlabel(r'$\varphi$'); plt.ylabel(r'$\theta$'); plt.title(r'$\theta(\varphi)$')
plt.show()