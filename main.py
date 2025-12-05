# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

G = 6.67430e-11

def rk4_step(y, dt, deriv):
    k1 = deriv(y)
    k2 = deriv(y + 0.5*dt*k1)
    k3 = deriv(y + 0.5*dt*k2)
    k4 = deriv(y + dt*k3)
    return y + dt/6*(k1+2*k2+2*k3+k4)

def two_body_deriv(state, m1, m2):
    # state: [r1x,r1y,r2x,r2y,v1x,v1y,v2x,v2y]
    r1 = state[0:2]
    r2 = state[2:4]
    v1 = state[4:6]
    v2 = state[6:8]
    dr = r2 - r1
    dist = np.linalg.norm(dr)
    force_mag = G*m1*m2/dist**3
    a1 = force_mag * dr / m1
    a2 = -force_mag * dr / m2
    return np.concatenate([v1, v2, a1, a2])

def simulate_orbit(m1, m2, r1_0, r2_0, v1_0, v2_0, dt, steps):
    state = np.concatenate([r1_0, r2_0, v1_0, v2_0])
    traj1 = np.zeros((steps, 2))
    traj2 = np.zeros((steps, 2))
    for i in range(steps):
        traj1[i] = state[0:2]
        traj2[i] = state[2:4]
        state = rk4_step(state, dt, lambda s: two_body_deriv(s, m1, m2))
    return traj1, traj2

def experiment1():
    m1 = 1e24
    m2 = 1e24
    d = 1e7
    r1_0 = np.array([-d/2, 0.0])
    r2_0 = np.array([ d/2, 0.0])
    mu = G * (m1 + m2)
    v_rel = np.sqrt(mu / d)
    v1_0 = np.array([0.0,  v_rel/2])
    v2_0 = np.array([0.0, -v_rel/2])
    T = 2 * np.pi * np.sqrt(d**3 / mu)
    dt = T / 1000
    steps = int(2 * T / dt)
    traj1, traj2 = simulate_orbit(m1, m2, r1_0, r2_0, v1_0, v2_0, dt, steps)
    plt.figure()
    plt.plot(traj1[:,0], traj1[:,1], label='Body 1')
    plt.plot(traj2[:,0], traj2[:,1], label='Body 2')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    plt.legend()
    plt.title('Two-body orbit')
    plt.savefig('orbit_trajectory.png')
    plt.close()
    return T

def experiment2():
    m1 = 1e24
    m2 = 1e24
    rs = np.logspace(5, 9, 200)
    forces = G * m1 * m2 / rs**2
    plt.figure()
    plt.loglog(rs, forces)
    plt.xlabel('Separation r (m)')
    plt.ylabel('Force magnitude (N)')
    plt.title('Gravitational force vs distance')
    plt.grid(True, which='both')
    plt.savefig('force_vs_distance.png')
    plt.close()

def main():
    T = experiment1()
    experiment2()
    print('Answer:', T)

if __name__ == '__main__':
    main()

